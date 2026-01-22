#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "fmm_funcs.hpp"
#include "fmm_interactions/bve_vel.hpp"
#include "forcing_funcs.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "interp_funcs_impl.hpp"
#include "general_utils_impl.hpp"

struct departure_to_target {
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_x;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_y;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_z;
	Kokkos::View<double**, Kokkos::LayoutRight> vors;
	Kokkos::View<double**, Kokkos::LayoutRight> new_vors;
	Kokkos::View<double***, Kokkos::LayoutRight> passive_tracers;
	Kokkos::View<double***, Kokkos::LayoutRight> new_passive_tracers;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int degree;
	int offset;
	double omega;

	departure_to_target(Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, Kokkos::View<double**, Kokkos::LayoutRight>& dep_x_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& dep_y_, Kokkos::View<double**, Kokkos::LayoutRight>& dep_z_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& vors_, Kokkos::View<double**, Kokkos::LayoutRight>& new_vors_, 
						Kokkos::View<double***, Kokkos::LayoutRight>& passive_tracers_, Kokkos::View<double***, Kokkos::LayoutRight>& new_passive_tracers_,
						Kokkos::View<CubedSpherePanel*> cubed_sphere_panels_, int degree_, int offset_, double omega_) : 
						zcos(zcos_), dep_x(dep_x_), dep_y(dep_y_), dep_z(dep_z_), vors(vors_), new_vors(new_vors_), passive_tracers(passive_tracers_), 
						new_passive_tracers(new_passive_tracers_), cubed_sphere_panels(cubed_sphere_panels_), degree(degree_), offset(offset_), omega(omega_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double bli_basis_vals[121];
		double xi, eta, xieta[2], vor;
		int panel_index, one_d_index;
		for (int j = 0; j < (degree+1)*(degree+1); j++) { // loop over points in each panel
			xieta_from_xyz(dep_x(i,j), dep_y(i,j), dep_z(i,j), xieta);
			panel_index = point_locate_panel(cubed_sphere_panels, dep_x(i,j), dep_y(i,j), dep_z(i,j));
			interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_xi, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_xi, 
							Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_eta, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_eta, degree);
			new_vors(i,j) = 0;
			for (int k = 0; k < passive_tracers.extent_int(2); k++) {
				new_passive_tracers(i,j,k) = 0;
			}
			vor = 0;
			one_d_index = 0;
			for (int m = 0; m < degree+1; m++) { // xi loop
				for (int n = 0; n < degree+1; n++) { // eta loop
					// one_d_index = m * (degree + 1) + n;
					vor += vors(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					for (int k = 0; k < passive_tracers.extent_int(2); k++) {
						new_passive_tracers(i,j,k) += passive_tracers(panel_index - offset, one_d_index, k) * bli_basis_vals[one_d_index];
					}
					one_d_index += 1;
				}
			}	
			vor += 2 * omega * dep_z(i,j);
			new_vors(i,j) = vor - 2*omega*zcos(i,j);
		}
	}
};

struct disp_update {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z;
	Kokkos::View<double**, Kokkos::LayoutRight> vor;
	Kokkos::View<double**, Kokkos::LayoutRight> new_vor;
	Kokkos::View<double**, Kokkos::LayoutRight> disp_x;
	Kokkos::View<double**, Kokkos::LayoutRight> disp_y;
	Kokkos::View<double**, Kokkos::LayoutRight> disp_z;
	double dt;
	double omega;
	int offset;

	disp_update(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vor_, Kokkos::View<double**, Kokkos::LayoutRight>& new_vor_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_x_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& disp_y_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_z_, double dt_, double omega_, int offset_) : xcos(xcos_), 
				ycos(ycos_), zcos(zcos_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), disp_x(disp_x_), disp_y(disp_y_), disp_z(disp_z_), vor(vor_), new_vor(new_vor_), dt(dt_), omega(omega_), offset(offset_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double x, y, z, dz;
		for (int j = 0; j < xcos.extent_int(1); j++) {
			x = xcos(i,j) + vel_x(i,j) * dt;
			y = ycos(i,j) + vel_y(i,j) * dt;
			z = zcos(i,j) + vel_z(i,j) * dt;
			project_to_sphere(x, y, z);
			new_vor(i,j) = vor(i,j) + 2 * omega * zcos(i,j) - 2 * omega * z;
			disp_x(i+offset,j) = dt * vel_x(i,j);
			disp_y(i+offset,j) = dt * vel_y(i,j);
			disp_z(i+offset,j) = dt * vel_z(i,j);
		}
	}
};

struct disp_interp {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z;
	Kokkos::View<double**, Kokkos::LayoutRight> disp_x;
	Kokkos::View<double**, Kokkos::LayoutRight> disp_y;
	Kokkos::View<double**, Kokkos::LayoutRight> disp_z;
	Kokkos::View<double**, Kokkos::LayoutRight> interp_vals;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	double dt;
	double omega;
	int offset;
	int degree;

	disp_interp(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& disp_x_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_z_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals_, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, double dt_, double omega_, int offset_, int degree_) : 
				xcos(xcos_), ycos(ycos_), zcos(zcos_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), disp_x(disp_x_), disp_y(disp_y_), disp_z(disp_z_), interp_vals(interp_vals_), 
				cubed_sphere_panels(cubed_sphere_panels_), dt(dt_), omega(omega_), offset(offset_), degree(degree_) {}

	void operator()(const int i) const {
		double xi_off, xi_scale, eta_off, eta_scale, x, y, z, xi, eta, xyz[3], bli_basis_vals[121];
		xi_off = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(i).max_xi + cubed_sphere_panels(i).min_xi);
		xi_scale = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(i).max_xi - cubed_sphere_panels(i).min_xi);
		eta_off = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(i).max_eta + cubed_sphere_panels(i).min_eta);
		eta_scale = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(i).max_eta - cubed_sphere_panels(i).min_eta);
		int panel_index, one_d_index;
		for (int j = 0; j < interp_vals.extent_int(0); j++) {
			xi = interp_vals(j,0) * xi_scale + xi_off;
			eta = interp_vals(j,1) * eta_scale + eta_off;
			xyz_from_xieta(xi, eta, cubed_sphere_panels(i).face, xyz);
			panel_index = point_locate_panel(cubed_sphere_panels, xyz[0], xyz[1], xyz[2]);
			interp_vals_bli(bli_basis_vals, xi, eta, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_xi, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_xi, 
							Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_eta, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_eta, degree);
			disp_x(i,j) = 0;
			disp_y(i,j) = 0;
			disp_z(i,j) = 0;
			one_d_index = 0;
			for (int k = 0; k < degree+1; k++) {
				for (int l = 0; l < degree+1; l++) {
					// one_d_index = k * (degree + 1) + l;
					disp_x(i,j) += disp_x(panel_index, one_d_index) * bli_basis_vals[one_d_index];
					disp_y(i,j) += disp_y(panel_index, one_d_index) * bli_basis_vals[one_d_index];
					disp_z(i,j) += disp_z(panel_index, one_d_index) * bli_basis_vals[one_d_index];
					one_d_index += 1;
				}
			}
		}
	}
};

void displacement_upward_pass(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
								Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
								Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, Kokkos::View<double**, Kokkos::LayoutRight>& vor, Kokkos::View<double**, Kokkos::LayoutRight>& new_vor, 
								Kokkos::View<double**, Kokkos::LayoutRight>& disp_x, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y, Kokkos::View<double**, Kokkos::LayoutRight>& disp_z, 
								Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, double dt, double omega) {
	int offset = cubed_sphere_panels.extent_int(0) - run_config.active_panel_count;
	Kokkos::parallel_for(run_config.active_panel_count, disp_update(xcos, ycos, zcos, vel_x, vel_y, vel_z, vor, new_vor, disp_x, disp_y, disp_z, dt, omega, offset));
	Kokkos::parallel_for(offset, disp_interp(xcos, ycos, zcos, vel_x, vel_y, vel_z, disp_x, disp_y, disp_z, interp_vals, cubed_sphere_panels, dt, omega, offset, run_config.interp_degree));
}

struct find_departure_points {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_0;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_0;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_0;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_2;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_2;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_2;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_3;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_3;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_3;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_x;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_y;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_z;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	double dt;
	int offset;
	int degree;

	find_departure_points(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_0_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_0_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_0_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_1_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_1_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_1_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_2_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_2_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_2_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_3_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_3_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_3_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& dep_x_, Kokkos::View<double**, Kokkos::LayoutRight>& dep_y_, Kokkos::View<double**, Kokkos::LayoutRight>& dep_z_, 
							Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, double dt_, int offset_, int degree_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), vel_x_0(vel_x_0_), vel_y_0(vel_y_0_), vel_z_0(vel_z_0_), vel_x_1(vel_x_1_), vel_y_1(vel_y_1_), vel_z_1(vel_z_1_), 
							vel_x_2(vel_x_2_), vel_y_2(vel_y_2_), vel_z_2(vel_z_2_), vel_x_3(vel_x_3_), vel_y_3(vel_y_3_), vel_z_3(vel_z_3_), dep_x(dep_x_), dep_y(dep_y_), dep_z(dep_z_), cubed_sphere_panels(cubed_sphere_panels_), dt(dt_), offset(offset_), degree(degree_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double k1x, k1y, k1z, x, y, z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
		int panel_index, one_d_index;
		double bli_basis_vals[121], xieta[2];
		for (int j = 0; j < xcos.extent_int(1); j++) {
			k1x = vel_x_3(i,j);
			k1y = vel_y_3(i,j);
			k1z = vel_z_3(i,j);

			// k2
			x = xcos(i,j) - 0.5*dt*k1x;
			y = ycos(i,j) - 0.5*dt*k1y;
			z = zcos(i,j) - 0.5*dt*k1z;
			project_to_sphere(x, y, z);
			xieta_from_xyz(x, y, z, xieta);
			panel_index = point_locate_panel(cubed_sphere_panels, x, y, z);
			interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_xi, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_xi, 
							Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_eta, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_eta, degree);

			k2x = 0;
			k2y = 0;
			k2z = 0;
			one_d_index = 0;
			for (int m = 0; m < degree+1; m++) { // xi loop
				for (int n = 0; n < degree+1; n++) { // eta loop
					// one_d_index = m * (degree + 1) + n;
					k2x += vel_x_2(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					k2y += vel_y_2(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					k2z += vel_z_2(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					one_d_index += 1;
				}
			}	

			// k3
			x = xcos(i,j) - 0.5*dt*k2x;
			y = ycos(i,j) - 0.5*dt*k2y;
			z = zcos(i,j) - 0.5*dt*k2z;
			project_to_sphere(x, y, z);
			xieta_from_xyz(x, y, z, xieta);
			panel_index = point_locate_panel(cubed_sphere_panels, x, y, z);
			interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_xi, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_xi, 
							Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_eta, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_eta, degree);

			k3x = 0;
			k3y = 0;
			k3z = 0;
			one_d_index = 0;
			for (int m = 0; m < degree+1; m++) { // xi loop
				for (int n = 0; n < degree+1; n++) { // eta loop
					// one_d_index = m * (degree + 1) + n;
					k3x += vel_x_1(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					k3y += vel_y_1(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					k3z += vel_z_1(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					one_d_index += 1;
				}
			}	

			// k4
			x = xcos(i,j) - dt*k3x;
			y = ycos(i,j) - dt*k3y;
			z = zcos(i,j) - dt*k3z;
			project_to_sphere(x, y, z);
			xieta_from_xyz(x, y, z, xieta);
			panel_index = point_locate_panel(cubed_sphere_panels, x, y, z);
			interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_xi, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_xi, 
							Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_eta, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_eta, degree);

			k4x = 0;
			k4y = 0;
			k4z = 0;
			one_d_index = 0;
			for (int m = 0; m < degree+1; m++) { // xi loop
				for (int n = 0; n < degree+1; n++) { // eta loop
					// one_d_index = m * (degree + 1) + n;
					k4x += vel_x_0(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					k4y += vel_y_0(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					k4z += vel_z_0(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					one_d_index += 1;
				}
			}	

			// combine them
			x = xcos(i,j) - 1.0/6.0*dt*k1x;
			y = ycos(i,j) - 1.0/6.0*dt*k1y;
			z = zcos(i,j) - 1.0/6.0*dt*k1z;
			project_to_sphere(x, y, z);
			x -= -1.0/3.0 * dt * k2x;
			y -= -1.0/3.0 * dt * k2y;
			z -= -1.0/3.0 * dt * k2z;
			project_to_sphere(x, y, z);
			x -= -1.0/3.0 * dt * k3x;
			y -= -1.0/3.0 * dt * k3y;
			z -= -1.0/3.0 * dt * k3z;
			project_to_sphere(x, y, z);
			x -= -1.0/6.0 * dt * k4x;
			y -= -1.0/6.0 * dt * k4y;
			z -= -1.0/6.0 * dt * k4z;
			project_to_sphere(x, y, z);
			dep_x(i,j) = x;
			dep_y(i,j) = y;
			dep_z(i,j) = z;
		}
	}
};

void bve_back_rk4_step(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& ycos, Kokkos::View<double**, Kokkos::LayoutRight>& zcos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& area, Kokkos::View<double**, Kokkos::LayoutRight>& vors, 
							Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_vors, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
							Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, Kokkos::View<double***, Kokkos::LayoutRight>& passive_tracers, double time) {
	// Compute v0
	// first do upward pass
	int dim2size = (run_config.interp_degree+1)*(run_config.interp_degree+1);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	Kokkos::View<double**, Kokkos::LayoutRight> effective_vorticity ("effective vorticities", run_config.active_panel_count, dim2size);
	bve_forcing(run_config, xcos, ycos, zcos, vors, effective_vorticity, time);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, effective_vorticity, proxy_source_vors);
	

	Kokkos::View<double**, Kokkos::LayoutRight> disp_x ("x displacements", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> disp_y ("y displacements", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> disp_z ("z displacements", run_config.panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {disp_x.extent_int(0), disp_x.extent_int(1)}), zero_out(disp_x));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {disp_y.extent_int(0), disp_y.extent_int(1)}), zero_out(disp_y));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {disp_z.extent_int(0), disp_z.extent_int(1)}), zero_out(disp_z));

	// next do velocity computation
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_1("proxy target vels 1", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_2("proxy target vels 1", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_3("proxy target vels 1", run_config.panel_count, dim2size);
	bve_vel_interactions(run_config, disp_x, disp_y, disp_z, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, interaction_list, cubed_sphere_panels, interp_vals);

	// downward pass
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_x.extent_int(0), vel_x.extent_int(1)}), zero_out(vel_x));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_y.extent_int(0), vel_y.extent_int(1)}), zero_out(vel_y));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_z.extent_int(0), vel_z.extent_int(1)}), zero_out(vel_z));
	downward_pass_3(run_config, interp_vals, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x, vel_y, vel_z);

	MPI_Allreduce(MPI_IN_PLACE, &vel_x(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// compute v1
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_1.extent_int(0), proxy_target_1.extent_int(1)}), zero_out(proxy_target_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_2.extent_int(0), proxy_target_2.extent_int(1)}), zero_out(proxy_target_2));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_3.extent_int(0), proxy_target_3.extent_int(1)}), zero_out(proxy_target_3));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	
	Kokkos::View<double**, Kokkos::LayoutRight> disp_vors ("vor after movement", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_1 ("vel x 1", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_1 ("vel y 1", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_1 ("vel z 1", run_config.active_panel_count, dim2size);

	displacement_upward_pass(run_config, xcos, ycos, zcos, vel_x, vel_y, vel_z, vors, disp_vors, disp_x, disp_y, disp_z, cubed_sphere_panels, interp_vals, 0.5*run_config.delta_t, run_config.omega);
	bve_forcing(run_config, xcos, ycos, zcos, disp_vors, effective_vorticity, time);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, effective_vorticity, proxy_source_vors);
	bve_vel_interactions(run_config, disp_x, disp_y, disp_z, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, interaction_list, cubed_sphere_panels, interp_vals);
	downward_pass_3(run_config, interp_vals, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x_1, vel_y_1, vel_z_1);
	MPI_Allreduce(MPI_IN_PLACE, &vel_x_1(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y_1(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z_1(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// compute v2
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_2 ("vel x 2", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_2 ("vel y 2", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_2 ("vel z 2", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_1.extent_int(0), proxy_target_1.extent_int(1)}), zero_out(proxy_target_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_2.extent_int(0), proxy_target_2.extent_int(1)}), zero_out(proxy_target_2));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_3.extent_int(0), proxy_target_3.extent_int(1)}), zero_out(proxy_target_3));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	displacement_upward_pass(run_config, xcos, ycos, zcos, vel_x_1, vel_y_1, vel_z_1, vors, disp_vors, disp_x, disp_y, disp_z, cubed_sphere_panels, interp_vals, 0.5*run_config.delta_t, run_config.omega);
	bve_forcing(run_config, xcos, ycos, zcos, disp_vors, effective_vorticity, time);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, effective_vorticity, proxy_source_vors);
	bve_vel_interactions(run_config, disp_x, disp_y, disp_z, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, interaction_list, cubed_sphere_panels, interp_vals);
	downward_pass_3(run_config, interp_vals, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x_2, vel_y_2, vel_z_2);
	MPI_Allreduce(MPI_IN_PLACE, &vel_x_2(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y_2(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z_2(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// compute v3
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_3 ("vel x 3", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_3 ("vel y 3", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_3 ("vel z 3", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_1.extent_int(0), proxy_target_1.extent_int(1)}), zero_out(proxy_target_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_2.extent_int(0), proxy_target_2.extent_int(1)}), zero_out(proxy_target_2));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_3.extent_int(0), proxy_target_3.extent_int(1)}), zero_out(proxy_target_3));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	displacement_upward_pass(run_config, xcos, ycos, zcos, vel_x_2, vel_y_2, vel_z_2, vors, disp_vors, disp_x, disp_y, disp_z, cubed_sphere_panels, interp_vals, run_config.delta_t, run_config.omega);
	bve_forcing(run_config, xcos, ycos, zcos, disp_vors, effective_vorticity, time);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, effective_vorticity, proxy_source_vors);
	bve_vel_interactions(run_config, disp_x, disp_y, disp_z, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, interaction_list, cubed_sphere_panels, interp_vals);
	downward_pass_3(run_config, interp_vals, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x_3, vel_y_3, vel_z_3);
	MPI_Allreduce(MPI_IN_PLACE, &vel_x_3(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y_3(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z_3(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// backwards RK4 to find departure points
	Kokkos::View<double**, Kokkos::LayoutRight> dep_x ("departure points x", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> dep_y ("departure points y", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> dep_z ("departure points z", run_config.active_panel_count, dim2size);
	int offset = cubed_sphere_panels.extent_int(0) - run_config.active_panel_count;
	Kokkos::parallel_for(xcos.extent_int(0), find_departure_points(xcos, ycos, zcos, vel_x, vel_y, vel_z, vel_x_1, vel_y_1, vel_z_1, vel_x_2, vel_y_2, vel_z_2, vel_x_3, vel_y_3, vel_z_3, dep_x, dep_y, dep_z, cubed_sphere_panels, run_config.delta_t, offset, run_config.interp_degree));
	
	// compute update from departure points to target points
	Kokkos::View<double**, Kokkos::LayoutRight> new_vors ("new vorticities", run_config.active_panel_count, dim2size);
	Kokkos::View<double***, Kokkos::LayoutRight> new_passive_tracers("new passive tarcers", run_config.active_panel_count, dim2size, run_config.tracer_count);
	
	Kokkos::parallel_for(run_config.active_panel_count, departure_to_target(
		zcos, dep_x, dep_y, dep_z, vors, new_vors, passive_tracers, new_passive_tracers, cubed_sphere_panels, run_config.interp_degree, offset, run_config.omega));

	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {vors.extent_int(0), vors.extent_int(1)}), copy_kokkos_view_2(vors, new_vors));
	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0, 0}, {passive_tracers.extent_int(0), passive_tracers.extent_int(1), passive_tracers.extent_int(2)}), copy_kokkos_view_3(passive_tracers, new_passive_tracers));
}
