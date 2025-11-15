#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "deriv_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "interp_funcs_impl.hpp"

struct departure_points {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_x;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_y;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_z;
	Kokkos::View<double**[3][3], Kokkos::LayoutRight> vel_jac;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int degree;
	int offset;
	double dt; 

	departure_points(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_,
						Kokkos::View<double**, Kokkos::LayoutRight>& dep_x_, Kokkos::View<double**, Kokkos::LayoutRight>& dep_y_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& dep_z_, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, 
						Kokkos::View<double**[3][3], Kokkos::LayoutRight>& vel_jac_, int degree_, int offset_, double dt_) : xcos(xcos_), 
						ycos(ycos_), zcos(zcos_), vel_x(vel_x_), vel_y(vel_y_),  vel_z(vel_z_), dep_x(dep_x_), dep_y(dep_y_), dep_z(dep_z_), 
						cubed_sphere_panels(cubed_sphere_panels_), vel_jac(vel_jac_), degree(degree_), offset(offset_), dt(dt_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		// computes the departure points for all the points in panel i
		double guess_x, guess_y, guess_z, tx, ty, tz;
		double bli_basis_vals[121];
		bool converged;
		double xi, eta, xieta[2], updatenorm, pointnorm;
		int panel_index, one_d_index;
		double matrix_buf[9], work_buf[21], rhs[3], sol[3];
		Kokkos::View<double**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> own_jac(matrix_buf, 3, 3);
		Kokkos::View<double**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> own_temp(work_buf, 3, 7);
		Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>> own_rhs(rhs, 3);
		Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>> own_sol(sol, 3);
		for (int j = 0; j < (degree+1)*(degree+1); j++) {
			// loop over points in panel
			// initial guess is just (x,y,z)-dt*(vx,vy,vz)
			guess_x = xcos(i,j) - dt*vel_x(i,j);
			guess_y = ycos(i,j) - dt*vel_y(i,j);
			guess_z = zcos(i,j) - dt*vel_z(i,j);
			tx = xcos(i,j);
			ty = ycos(i,j);
			tz = zcos(i,j); // target points
			converged = false;
			while (not converged) {
				// convert (gx, gy, gz) to xi eta coords
				xieta_from_xyz(guess_x, guess_y, guess_z, xieta);
				// find which panel (gx, gy, gz) is in
				panel_index = point_locate_panel(cubed_sphere_panels, guess_x, guess_y, guess_z);
				// interpolate jacobian and RHS of newton solve at (gx, gy, gz)
				interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], M_PI/4.0*cubed_sphere_panels(panel_index).min_xi, M_PI/4.0*cubed_sphere_panels(panel_index).max_xi, 
								M_PI/4.0*cubed_sphere_panels(panel_index).min_eta, M_PI/4.0*cubed_sphere_panels(panel_index).max_eta, degree);
				for (int k = 0; k < 3; k++) { // jac first index loop
					for (int l = 0; l < 3; l++) { // jac second index loop
						own_jac(k,l) = 0.0;
					}
				}
				own_rhs(0) = 0.0;
				own_rhs(1) = 0.0;
				own_rhs(2) = 0.0;

				for (int m = 0; m < degree+1; m++) { // xi loop
					for (int n = 0; n < degree+1; n++) { // eta loop
						one_d_index = m * (degree + 1) + n;
						for (int k = 0; k < 3; k++) { // jac first index
							for (int l = 0; l < 3; l++) { // jac second index
								own_jac(k,l) += vel_jac(panel_index - offset,one_d_index,k,l) * bli_basis_vals[one_d_index];
							}
						}
						own_rhs(0) += vel_x(panel_index - offset,one_d_index) * bli_basis_vals[one_d_index];
						own_rhs(1) += vel_y(panel_index - offset,one_d_index) * bli_basis_vals[one_d_index];
						own_rhs(2) += vel_z(panel_index - offset,one_d_index) * bli_basis_vals[one_d_index];
					}
				}	

				own_rhs(0) *= dt;
				own_rhs(1) *= dt;
				own_rhs(2) *= dt;
				own_rhs(0) += guess_x - tx;
				own_rhs(1) += guess_y - ty;
				own_rhs(2) += guess_z - tz;

				for (int k = 0; k < 3; k++) { // jac first index loop
					for (int l = 0; l < 3; l++) { // jac second index loop
						own_jac(k,l) *= dt;
					}
					own_jac(k,k) += 1.0;
				}	

				// solve linear system
				KokkosBatched::SerialGesv<KokkosBatched::Gesv::StaticPivoting>::invoke(own_jac, own_sol, own_rhs, own_temp);

				// own_sol is the update
				guess_x += own_sol(0);
				guess_y += own_sol(1);
				guess_z += own_sol(2);

				pointnorm = Kokkos::sqrt(guess_x*guess_x + guess_y*guess_y + guess_z*guess_z);
				guess_x /= pointnorm;
				guess_y /= pointnorm;
				guess_z /= pointnorm; // project to sphere

				// check size of update
				updatenorm = Kokkos::sqrt(own_sol(0)*own_sol(0)+own_sol(1)*own_sol(1)+own_sol(2)*own_sol(2));
				if (updatenorm < 1e-14) {
					converged = true; // stop iteration
					dep_x(i,j) = guess_x;
					dep_y(i,j) = guess_y;
					dep_z(i,j) = guess_z;
				}
			}
		}
	}
};