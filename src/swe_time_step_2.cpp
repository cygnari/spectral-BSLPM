#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "fmm_funcs.hpp"
#include "fmm_interactions/swe_vel.hpp"
#include "forcing_funcs.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "interp_funcs_impl.hpp"
#include "general_utils_impl.hpp"
#include "deriv_funcs_impl.hpp"
#include "deriv_funcs.hpp"
#include "fmm_interactions/laplacian.hpp"

struct swe_departure_to_target_2 {
	Kokkos::View<double**, Kokkos::LayoutRight> dep_x;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_y;
	Kokkos::View<double**, Kokkos::LayoutRight> dep_z;
	Kokkos::View<double**, Kokkos::LayoutRight> arrival_vor;
	Kokkos::View<double**, Kokkos::LayoutRight> new_vors;
	Kokkos::View<double***, Kokkos::LayoutRight> passive_tracers;
	Kokkos::View<double***, Kokkos::LayoutRight> new_passive_tracers;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int degree;
	int offset;
	double omega;

	swe_departure_to_target_2(Kokkos::View<double**, Kokkos::LayoutRight>& dep_x_, Kokkos::View<double**, Kokkos::LayoutRight>& dep_y_, Kokkos::View<double**, Kokkos::LayoutRight>& dep_z_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& arrival_vor_,  Kokkos::View<double**, Kokkos::LayoutRight>& new_vors_, 
						Kokkos::View<double***, Kokkos::LayoutRight>& passive_tracers_, Kokkos::View<double***, Kokkos::LayoutRight>& new_passive_tracers_,
						Kokkos::View<CubedSpherePanel*> cubed_sphere_panels_, int degree_, int offset_, double omega_) : dep_x(dep_x_), dep_y(dep_y_), dep_z(dep_z_), arrival_vor(arrival_vor_), new_vors(new_vors_), 
						passive_tracers(passive_tracers_), new_passive_tracers(new_passive_tracers_), cubed_sphere_panels(cubed_sphere_panels_), degree(degree_), offset(offset_), omega(omega_) {}

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
			one_d_index = 0;
			for (int m = 0; m < degree+1; m++) { // xi loop
				for (int n = 0; n < degree+1; n++) { // eta loop
					new_vors(i,j) += arrival_vor(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					for (int k = 0; k < passive_tracers.extent_int(2); k++) {
						new_passive_tracers(i,j,k) += passive_tracers(panel_index - offset, one_d_index, k) * bli_basis_vals[one_d_index];
					}
					one_d_index += 1;
				}
			}	
		}
	}
};

struct swe_tendency_computation_2{
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z;
	Kokkos::View<double**, Kokkos::LayoutRight> vor;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend;
	Kokkos::View<double**, Kokkos::LayoutRight> div;
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend;
	Kokkos::View<double**, Kokkos::LayoutRight> height;
	Kokkos::View<double**, Kokkos::LayoutRight> height_tend;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	double omega;
	int degree;
	int offset;

	swe_tendency_computation_2(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vor_, Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend_, Kokkos::View<double**, Kokkos::LayoutRight>& div_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& div_tend_, Kokkos::View<double**, Kokkos::LayoutRight>& height_,  Kokkos::View<double**, Kokkos::LayoutRight>& height_tend_, 
				Kokkos::View<CubedSpherePanel*> cubed_sphere_panels_, double omega_, int degree_, int offset_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), 
				vor(vor_), vor_tend(vor_tend_), div(div_), div_tend(div_tend_), height(height_), height_tend(height_tend_), cubed_sphere_panels(cubed_sphere_panels_), omega(omega_), degree(degree_), offset(offset_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double min_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i+offset).min_xi;
		double max_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i+offset).max_xi;
		double min_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i+offset).min_eta;
		double max_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i+offset).max_eta;
		double xi_vel[121], eta_vel[121], curl_rad_comp[121], lap_comp[121], dp, height_lap[121], div_comp_xi[121], div_comp_eta[121], height_div[121], zetaf;
		for (int j = 0; j < xcos.extent_int(1); j++) {
			xietavec_from_xyzvec(xi_vel[j], eta_vel[j], vel_x(i,j), vel_y(i,j), vel_z(i,j), xcos(i,j), ycos(i,j), zcos(i,j));
			zetaf = vor(i,j) + 2 * omega * zcos(i,j);
			xi_vel[j] *= zetaf;
			eta_vel[j] *= zetaf;
			dp = vel_x(i,j)*vel_x(i,j) + vel_y(i,j)*vel_y(i,j) + vel_z(i,j)*vel_z(i,j);
			lap_comp[j] = height(i,j) + 0.5*dp;
			div_comp_xi[j] = height(i,j) * xi_vel[j];
			div_comp_eta[j] = height(i,j) * eta_vel[j];
		}
		single_panel_curl_rad_comp(curl_rad_comp, xi_vel, eta_vel, min_xi, max_xi, min_eta, max_eta, degree);
		single_panel_lap(height_lap, lap_comp, min_xi, max_xi, min_eta, max_eta, degree);
		single_panel_div(height_div, div_comp_xi, div_comp_eta, min_xi, max_xi, min_eta, max_eta, degree);

		double abs_vor, abs_vor_tend, x, y, z, grad_comp, vel_u;
		double lat, lon;
		for (int j = 0; j < xcos.extent_int(1); j++) {
			x = xcos(i,j);
			y = ycos(i,j);
			z = zcos(i,j);
			xyz_to_latlon(lat, lon, x, y, z);
			abs_vor = vor(i,j) + 2.0 * omega * z;
			abs_vor_tend = -div(i,j) * abs_vor;
			vor_tend(i,j) = abs_vor_tend - 2.0 * omega * vel_z(i,j);
			
			// div_tend(i,j) = 2.0*omega*zcos(i,j)*vor(i,j); // f * zeta
			div_tend(i,j) = -curl_rad_comp[j] - height_lap[j];
			height_tend(i,j) = -height_div[j];

			// // grad_comp = vel_grad_11[j]*vel_grad_11[j] + vel_grad_12[j]*vel_grad_21[j] + vel_grad_13[j]*vel_grad_31[j] + 
			// // 			vel_grad_21[j]*vel_grad_12[j] + vel_grad_22[j]*vel_grad_22[j] + vel_grad_23[j]*vel_grad_32[j] + 
			// // 			vel_grad_31[j]*vel_grad_13[j] + vel_grad_32[j]*vel_grad_23[j] + vel_grad_33[j]*vel_grad_33[j];
			// // div_tend(i,j) -= grad_comp;
			// div_tend(i,j) += 2.0 * (M_PI/(6.0*86400.0)*M_PI/(6.0*86400.0)) * Kokkos::sin(lat) * Kokkos::sin(lat); // grad comp

			// // div_tend(i,j) -= (M_PI/(6.0*86400.0) + 2.0 * omega) * (M_PI/(6.0*86400.0)) * (2.0*Kokkos::sin(lat)*Kokkos::sin(lat)-Kokkos::cos(lat)*Kokkos::cos(lat)); // height lap
			// div_tend(i,j) -= height_lap(i,j);

			// dp = vel_x(i,j)*vel_x(i,j) + vel_y(i,j)*vel_y(i,j) + vel_z(i,j)*vel_z(i,j);
			// div_tend(i,j) -= dp;
			
			// vel_u = -y*vel_x(i,j) + x*vel_y(i,j);
			// if ((x*x+y*y) > 1e-16) {
			// 	div_tend(i,j) -= 2.0 * omega * vel_u; // grad f term
			// }
			if ((i == 0) and (j == 0)) {
				std::cout << div_tend(i,j) << std::endl;
				std::cout << lat << " " << lon << std::endl;
				std::cout << "Curl term: " << -curl_rad_comp[j] << std::endl;
				// std::cout << "u dot u: " << -dp << std::endl;
				std::cout << "height lap: " << -height_lap[j] << std::endl;
				std::cout << height_tend(i,j) << std::endl;
				// std::cout << "double dot prod: " << grad_comp << std::endl;
				// std::cout << "f*zeta: " << 2.0*omega*zcos(i,j)*vor(i,j) << std::endl;
			}
			// // div_tend(i,j) = 0.0;
			// height_tend(i,j) = -height(i,j) * div(i,j);
		}
	}
};

struct swe_apply_tendencies_2 {
	Kokkos::View<double**, Kokkos::LayoutRight> vor;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend;
	Kokkos::View<double**, Kokkos::LayoutRight> new_vor;
	Kokkos::View<double**, Kokkos::LayoutRight> div;
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend;
	Kokkos::View<double**, Kokkos::LayoutRight> new_div;
	Kokkos::View<double**, Kokkos::LayoutRight> height;
	Kokkos::View<double**, Kokkos::LayoutRight> height_tend;
	Kokkos::View<double**, Kokkos::LayoutRight> new_height;
	Kokkos::View<double**, Kokkos::LayoutRight> areas;
	Kokkos::View<double**, Kokkos::LayoutRight> new_areas;
	double dt;

	swe_apply_tendencies_2(Kokkos::View<double**, Kokkos::LayoutRight>& vor_, Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend_, Kokkos::View<double**, Kokkos::LayoutRight>& new_vor_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& div_, Kokkos::View<double**, Kokkos::LayoutRight>& div_tend_, Kokkos::View<double**, Kokkos::LayoutRight>& new_div_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& height_, Kokkos::View<double**, Kokkos::LayoutRight>& height_tend_, Kokkos::View<double**, Kokkos::LayoutRight>& new_height_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& areas_, Kokkos::View<double**, Kokkos::LayoutRight>& new_areas_, double dt_) : vor(vor_), vor_tend(vor_tend_), new_vor(new_vor_), 
						div(div_), div_tend(div_tend_), new_div(new_div_), height(height_), height_tend(height_tend_), new_height(new_height_), areas(areas_), new_areas(new_areas_), dt(dt_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = vor.extent_int(1);
		for (int j = 0; j < jmax; j++) {
			new_vor(i,j) = vor(i,j) + dt * vor_tend(i,j);
			new_div(i,j) = div(i,j) + dt * div_tend(i,j);
			new_height(i,j) = height(i,j) + dt * height_tend(i,j);
			new_areas(i,j) = areas(i,j) * (1.0 + dt * div(i,j));
		}
	}
};

struct swe_disp_update_2 {
	Kokkos::View<double**, Kokkos::LayoutRight> vel;
	Kokkos::View<double**, Kokkos::LayoutRight> disp;
	double dt;
	int offset;

	swe_disp_update_2(Kokkos::View<double**, Kokkos::LayoutRight>& vel_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_, double dt_, int offset_) : vel(vel_), disp(disp_), dt(dt_), offset(offset_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = vel.extent_int(1);
		for (int j = 0; j < jmax; j++) {
			disp(i+offset,j) = dt * vel(i,j);
		}
	}
};

struct swe_disp_interp_2 {
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
	int degree;

	swe_disp_interp_2(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& disp_x_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_z_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals_, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, double dt_, int degree_) : 
				xcos(xcos_), ycos(ycos_), zcos(zcos_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), disp_x(disp_x_), disp_y(disp_y_), disp_z(disp_z_), interp_vals(interp_vals_), 
				cubed_sphere_panels(cubed_sphere_panels_), dt(dt_), degree(degree_) {}

	KOKKOS_INLINE_FUNCTION
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
					disp_x(i,j) += disp_x(panel_index, one_d_index) * bli_basis_vals[one_d_index];
					disp_y(i,j) += disp_y(panel_index, one_d_index) * bli_basis_vals[one_d_index];
					disp_z(i,j) += disp_z(panel_index, one_d_index) * bli_basis_vals[one_d_index];
					one_d_index += 1;
				}
			}
		}
	}
};

void swe_displacement_upward_pass_2(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
								Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
								Kokkos::View<double**, Kokkos::LayoutRight>& vel_z,  
								Kokkos::View<double**, Kokkos::LayoutRight>& disp_x, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y, Kokkos::View<double**, Kokkos::LayoutRight>& disp_z, 
								Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, double dt) {
	int offset = cubed_sphere_panels.extent_int(0) - run_config.active_panel_count;
	Kokkos::parallel_for(run_config.active_panel_count, swe_disp_update_2(vel_x, disp_x, dt, offset));
	Kokkos::parallel_for(run_config.active_panel_count, swe_disp_update_2(vel_y, disp_y, dt, offset));
	Kokkos::parallel_for(run_config.active_panel_count, swe_disp_update_2(vel_z, disp_z, dt, offset));
	Kokkos::parallel_for(offset, swe_disp_interp_2(xcos, ycos, zcos, vel_x, vel_y, vel_z, disp_x, disp_y, disp_z, interp_vals, cubed_sphere_panels, dt, run_config.interp_degree));
}

struct swe_compute_arrival_values_2 {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vor;
	Kokkos::View<double**, Kokkos::LayoutRight> div;
	Kokkos::View<double**, Kokkos::LayoutRight> height;
	Kokkos::View<double**, Kokkos::LayoutRight> a_vor;
	Kokkos::View<double**, Kokkos::LayoutRight> a_div;
	Kokkos::View<double**, Kokkos::LayoutRight> a_height;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_2;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_2;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_2;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_3;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_3;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_3;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_4;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_4;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_4;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend_2;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend_3;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend_4;
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend_1;
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend_2;
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend_3;
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend_4;
	Kokkos::View<double**, Kokkos::LayoutRight> height_tend_1;
	Kokkos::View<double**, Kokkos::LayoutRight> height_tend_2;
	Kokkos::View<double**, Kokkos::LayoutRight> height_tend_3;
	Kokkos::View<double**, Kokkos::LayoutRight> height_tend_4;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	double dt;
	int degree;
	int offset;

	swe_compute_arrival_values_2(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_,
							Kokkos::View<double**, Kokkos::LayoutRight>& vor_, Kokkos::View<double**, Kokkos::LayoutRight>& div_, Kokkos::View<double**, Kokkos::LayoutRight>& height_,
							Kokkos::View<double**, Kokkos::LayoutRight>& a_vor_, Kokkos::View<double**, Kokkos::LayoutRight>& a_div_, Kokkos::View<double**, Kokkos::LayoutRight>& a_height_,
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_1_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_1_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_1_,
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_2_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_2_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_2_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_3_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_3_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_3_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_4_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_4_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_4_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend_1_, Kokkos::View<double**, Kokkos::LayoutRight>& div_tend_1_, Kokkos::View<double**, Kokkos::LayoutRight>& h_tend_1_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend_2_, Kokkos::View<double**, Kokkos::LayoutRight>& div_tend_2_, Kokkos::View<double**, Kokkos::LayoutRight>& h_tend_2_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend_3_, Kokkos::View<double**, Kokkos::LayoutRight>& div_tend_3_, Kokkos::View<double**, Kokkos::LayoutRight>& h_tend_3_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend_4_, Kokkos::View<double**, Kokkos::LayoutRight>& div_tend_4_, Kokkos::View<double**, Kokkos::LayoutRight>& h_tend_4_, 
							Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, double dt_, int degree_, int offset_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), vor(vor_), div(div_), height(height_), 
							vel_x_1(vel_x_1_), vel_y_1(vel_x_1_), vel_z_1(vel_x_1_), vel_x_2(vel_x_2_), vel_y_2(vel_x_2_), vel_z_2(vel_x_2_), vel_x_3(vel_x_3_), vel_y_3(vel_x_3_), vel_z_3(vel_x_3_),
							vel_x_4(vel_x_4_), vel_y_4(vel_x_4_), vel_z_4(vel_x_4_), vor_tend_1(vor_tend_1_), div_tend_1(div_tend_1_), height_tend_1(h_tend_1_), vor_tend_2(vor_tend_2_), div_tend_2(div_tend_2_), height_tend_2(h_tend_2_), 
							vor_tend_3(vor_tend_3_), div_tend_3(div_tend_3_), height_tend_3(h_tend_3_), vor_tend_4(vor_tend_4_), div_tend_4(div_tend_4_), height_tend_4(h_tend_4_), cubed_sphere_panels(cubed_sphere_panels_), 
							dt(dt_), degree(degree_), offset(offset_), a_vor(a_vor_), a_div(a_div_), a_height(a_height_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double vor1, vor2, vor3, vor4, x, y, z, xieta[2], bli_basis_vals[121];
		int panel_index, one_d_index;
		for (int j = 0; j < jmax; j++) {
			// k1
			vor1 = vor_tend_1(i,j);

			// k2
			x = xcos(i,j) + 0.5*dt*vel_x_1(i,j);
			y = ycos(i,j) + 0.5*dt*vel_y_1(i,j);
			z = zcos(i,j) + 0.5*dt*vel_z_1(i,j);
			project_to_sphere(x, y, z);
			xieta_from_xyz(x, y, z, xieta);
			panel_index = point_locate_panel(cubed_sphere_panels, x, y, z);
			interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_xi, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_xi, 
							Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_eta, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_eta, degree);
			vor2 = 0;
			one_d_index = 0;
			for (int m = 0; m < degree+1; m++) {
				for (int n = 0; n < degree+1; n++) {
					vor2 += vor_tend_2(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					one_d_index += 1;
				}
			}

			// k3
			x = xcos(i,j) + 0.5*dt*vel_x_2(i,j);
			y = ycos(i,j) + 0.5*dt*vel_y_2(i,j);
			z = zcos(i,j) + 0.5*dt*vel_z_2(i,j);
			project_to_sphere(x, y, z);
			xieta_from_xyz(x, y, z, xieta);
			panel_index = point_locate_panel(cubed_sphere_panels, x, y, z);
			interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_xi, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_xi, 
							Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_eta, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_eta, degree);
			vor3 = 0;
			one_d_index = 0;
			for (int m = 0; m < degree+1; m++) {
				for (int n = 0; n < degree+1; n++) {
					vor3 += vor_tend_3(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					one_d_index += 1;
				}
			}

			// k4
			x = xcos(i,j) + dt*vel_x_3(i,j);
			y = xcos(i,j) + dt*vel_y_3(i,j);
			z = xcos(i,j) + dt*vel_z_3(i,j);
			project_to_sphere(x, y, z);
			xieta_from_xyz(x, y, z, xieta);
			panel_index = point_locate_panel(cubed_sphere_panels, x, y, z);
			interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_xi, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_xi, 
							Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).min_eta, Kokkos::numbers::pi/4.0*cubed_sphere_panels(panel_index).max_eta, degree);
			vor4 = 0;
			one_d_index = 0;

			for (int m = 0; m < degree+1; m++) {
				for (int n = 0; n < degree+1; n++) {
					vor3 += vor_tend_4(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					one_d_index += 1;
				}
			}

			// combine them
			a_vor(i,j) = vor(i,j) + dt/6.0*(vor1 + 2.0*vor2 + 2.0*vor3 + vor4);
			a_div(i,j) = div(i,j) + dt/6.0*(div_tend_1(i,j) + 2.0*div_tend_2(i,j) + 2.0*div_tend_3(i,j) + div_tend_4(i,j));
			a_height(i,j) = height(i,j) + dt/6.0*(height_tend_1(i,j) + 2.0*height_tend_2(i,j) + 2.0*height_tend_3(i,j) + height_tend_4(i,j));
			// a_vor(i,j) = vor(i,j) + dt*vor1;
			// a_div(i,j) = div(i,j) + dt*div1;
			// a_height(i,j) = height(i,j) + dt*h1;
		}
	}
};

struct swe_find_departure_points_2 {
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

	swe_find_departure_points_2(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
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

void swe_back_rk4_step_2(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& area, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vors, Kokkos::View<double**, Kokkos::LayoutRight>& divs, Kokkos::View<double**, 
							Kokkos::LayoutRight>& height, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
							Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, Kokkos::View<double***, Kokkos::LayoutRight>& passive_tracers, Kokkos::View<double**, Kokkos::LayoutRight>& height_lapl, double time) {
	int offset = cubed_sphere_panels.extent_int(0) - run_config.active_panel_count;
	int dim2size = (run_config.interp_degree+1)*(run_config.interp_degree+1);
	// Compute k1
	// first do upward pass
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_vors ("proxy source vors", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_divs ("proxy source divs", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_heights ("proxy source heights", run_config.panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_divs.extent_int(0), proxy_source_divs.extent_int(1)}), zero_out(proxy_source_divs));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_heights.extent_int(0), proxy_source_heights.extent_int(1)}), zero_out(proxy_source_heights));
	// Kokkos::View<double**, Kokkos::LayoutRight> effective_vorticity ("effective vorticities", run_config.active_panel_count, dim2size);
	
	// laplacian of g*s
	// Kokkos::View<double**, Kokkos::LayoutRight> height_lap_1 ("laplacian of free surface 1", run_config.active_panel_count, dim2size);
	// laplacian(run_config, height_lap_1, height, cubed_sphere_panels);
	// height_lapl = height_lap_1;

	// velocity computation
	// bve_forcing(run_config, xcos, ycos, zcos, vors, effective_vorticity, time);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, vors, proxy_source_vors);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, divs, proxy_source_divs);

	Kokkos::View<double**, Kokkos::LayoutRight> disp_x ("x displacements", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> disp_y ("y displacements", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> disp_z ("z displacements", run_config.panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {disp_x.extent_int(0), disp_x.extent_int(1)}), zero_out(disp_x));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {disp_y.extent_int(0), disp_y.extent_int(1)}), zero_out(disp_y));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {disp_z.extent_int(0), disp_z.extent_int(1)}), zero_out(disp_z));

	// next do velocity computation
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_1("proxy target vels", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_2("proxy target vels", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_3("proxy target vels", run_config.panel_count, dim2size);
	swe_vel_interactions_2(run_config, disp_x, disp_y, disp_z, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, proxy_source_divs, interaction_list, cubed_sphere_panels, interp_vals);

	// downward pass
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_x.extent_int(0), vel_x.extent_int(1)}), zero_out(vel_x));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_y.extent_int(0), vel_y.extent_int(1)}), zero_out(vel_y));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_z.extent_int(0), vel_z.extent_int(1)}), zero_out(vel_z));
	downward_pass_3(run_config, interp_vals, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x, vel_y, vel_z);

	MPI_Allreduce(MPI_IN_PLACE, &vel_x(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_x, 1e-9));
	// Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_y, 1e-9));
	Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_z, 1e-8));

	// compute tendencies
	Kokkos::View<double**, Kokkos::LayoutRight> vor_update_1 ("vorticity k1", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> div_update_1 ("divergence k1", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> height_update_1 ("height k1", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(run_config.active_panel_count, swe_tendency_computation_2(xcos, ycos, zcos, vel_x, vel_y, vel_z, vors, vor_update_1, divs, div_update_1, height, height_update_1, cubed_sphere_panels, run_config.omega,run_config.interp_degree, offset));

	// compute k2
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_1.extent_int(0), proxy_target_1.extent_int(1)}), zero_out(proxy_target_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_2.extent_int(0), proxy_target_2.extent_int(1)}), zero_out(proxy_target_2));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_3.extent_int(0), proxy_target_3.extent_int(1)}), zero_out(proxy_target_3));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_divs.extent_int(0), proxy_source_divs.extent_int(1)}), zero_out(proxy_source_divs));
	
	Kokkos::View<double**, Kokkos::LayoutRight> disp_vors ("vor after movement", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> disp_divs ("div after movement", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> disp_heights ("height after movement", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_2 ("vel x 2", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_2 ("vel y 2", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_2 ("vel z 2", run_config.active_panel_count, dim2size);
	// Kokkos::View<double**, Kokkos::LayoutRight> height_lap_2 ("laplacian of free surface 2", run_config.active_panel_count, dim2size);
	// Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_h("proxy target heights", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> disp_areas ("area after movement", run_config.active_panel_count, dim2size);

	swe_displacement_upward_pass_2(run_config, xcos, ycos, zcos, vel_x, vel_y, vel_z, disp_x, disp_y, disp_z, cubed_sphere_panels, interp_vals, 0.5*run_config.delta_t);
	Kokkos::parallel_for(run_config.active_panel_count, swe_apply_tendencies_2(vors, vor_update_1, disp_vors, divs, div_update_1, disp_divs, height, height_update_1, disp_heights, area, disp_areas, 0.5*run_config.delta_t));
	
	// bve_forcing(run_config, xcos, ycos, zcos, disp_vors, effective_vorticity, time);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, disp_areas, disp_vors, proxy_source_vors);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, disp_divs, proxy_source_divs);
	// upward_pass(run_config, interp_vals, cubed_sphere_panels, disp_areas, disp_heights, proxy_source_heights);
	swe_vel_interactions_2(run_config, disp_x, disp_y, disp_z, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, proxy_source_divs, interaction_list, cubed_sphere_panels, interp_vals);
	// laplacian_interactions(run_config, disp_x, disp_y, disp_z, proxy_target_h, proxy_source_heights, interaction_list, cubed_sphere_panels, interp_vals);
	downward_pass_3(run_config, interp_vals, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x_2, vel_y_2, vel_z_2);
	// downward_pass(run_config, interp_vals, cubed_sphere_panels, proxy_target_h, height_lap_2);

	MPI_Allreduce(MPI_IN_PLACE, &vel_x_2(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y_2(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z_2(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// MPI_Allreduce(MPI_IN_PLACE, &height_lap_2(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_x_2, 1e-9));
	// Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_y_2, 1e-9));
	Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_z_2, 1e-8));
	// Kokkos::parallel_for(run_config.active_panel_count, swe_height_correction(xcos, ycos, zcos, vel_x, vel_y, vel_z, height, height_update_1, cubed_sphere_panels, offset, 0.5*run_config.delta_t, run_config.interp_degree));
	// height_lapl = height_lap_2;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_update_2 ("vorticity k2", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> div_update_2 ("divergence k2", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> height_update_2 ("height k2", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(run_config.active_panel_count, swe_tendency_computation_2(xcos, ycos, zcos, vel_x_2, vel_y_2, vel_z_2, vors, vor_update_2, divs, div_update_2, height, height_update_2, cubed_sphere_panels, run_config.omega,run_config.interp_degree, offset));
	
	// // compute k3
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_3 ("vel x 3", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_3 ("vel y 3", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_3 ("vel z 3", run_config.active_panel_count, dim2size);
	// Kokkos::View<double**, Kokkos::LayoutRight> height_lap_3 ("laplacian of free surface 3", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_1.extent_int(0), proxy_target_1.extent_int(1)}), zero_out(proxy_target_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_2.extent_int(0), proxy_target_2.extent_int(1)}), zero_out(proxy_target_2));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_3.extent_int(0), proxy_target_3.extent_int(1)}), zero_out(proxy_target_3));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_h.extent_int(0), proxy_target_h.extent_int(1)}), zero_out(proxy_target_h));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_divs.extent_int(0), proxy_source_divs.extent_int(1)}), zero_out(proxy_source_divs));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_heights.extent_int(0), proxy_source_heights.extent_int(1)}), zero_out(proxy_source_heights));

	swe_displacement_upward_pass_2(run_config, xcos, ycos, zcos, vel_x_2, vel_y_2, vel_z_2, disp_x, disp_y, disp_z, cubed_sphere_panels, interp_vals, 0.5*run_config.delta_t);
	Kokkos::parallel_for(run_config.active_panel_count, swe_apply_tendencies_2(vors, vor_update_2, disp_vors, divs, div_update_2, disp_divs, height, height_update_2, disp_heights, area, disp_areas, 0.5*run_config.delta_t));

	// // bve_forcing(run_config, xcos, ycos, zcos, disp_vors, effective_vorticity, time);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, disp_areas, disp_vors, proxy_source_vors);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, disp_divs, proxy_source_divs);
	// upward_pass(run_config, interp_vals, cubed_sphere_panels, disp_areas, disp_heights, proxy_source_heights);
	swe_vel_interactions_2(run_config, disp_x, disp_y, disp_z, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, proxy_source_divs, interaction_list, cubed_sphere_panels, interp_vals);
	// laplacian_interactions(run_config, disp_x, disp_y, disp_z, proxy_target_h, proxy_source_heights, interaction_list, cubed_sphere_panels, interp_vals);
	downward_pass_3(run_config, interp_vals, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x_3, vel_y_3, vel_z_3);
	// downward_pass(run_config, interp_vals, cubed_sphere_panels, proxy_target_h, height_lap_3);

	MPI_Allreduce(MPI_IN_PLACE, &vel_x_3(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y_3(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z_3(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// MPI_Allreduce(MPI_IN_PLACE, &height_lap_3(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_x_3, 1e-9));
	// Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_y_3, 1e-9));
	Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_z_3, 1e-8));
	// Kokkos::parallel_for(run_config.active_panel_count, swe_height_correction(xcos, ycos, zcos, vel_x_2, vel_y_2, vel_z_2, height, height_update_2, height_lap_3, cubed_sphere_panels, offset, 0.5*run_config.delta_t, run_config.interp_degree));

	Kokkos::View<double**, Kokkos::LayoutRight> vor_update_3 ("vorticity k3", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> div_update_3 ("divergence k3", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> height_update_3 ("height k3", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(run_config.active_panel_count, swe_tendency_computation_2(xcos, ycos, zcos, vel_x_3, vel_y_3, vel_z_3, vors, vor_update_3, divs, div_update_3, height, height_update_3, cubed_sphere_panels, run_config.omega,run_config.interp_degree, offset));

	// // compute k4
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_4 ("vel x 4", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_4 ("vel y 4", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_4 ("vel z 4", run_config.active_panel_count, dim2size);
	// Kokkos::View<double**, Kokkos::LayoutRight> height_lap_4 ("laplacian of free surface 4", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_1.extent_int(0), proxy_target_1.extent_int(1)}), zero_out(proxy_target_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_2.extent_int(0), proxy_target_2.extent_int(1)}), zero_out(proxy_target_2));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_3.extent_int(0), proxy_target_3.extent_int(1)}), zero_out(proxy_target_3));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_h.extent_int(0), proxy_target_h.extent_int(1)}), zero_out(proxy_target_h));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_divs.extent_int(0), proxy_source_divs.extent_int(1)}), zero_out(proxy_source_divs));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_heights.extent_int(0), proxy_source_heights.extent_int(1)}), zero_out(proxy_source_heights));

	swe_displacement_upward_pass_2(run_config, xcos, ycos, zcos, vel_x_3, vel_y_3, vel_z_3, disp_x, disp_y, disp_z, cubed_sphere_panels, interp_vals, run_config.delta_t);
	Kokkos::parallel_for(run_config.active_panel_count, swe_apply_tendencies_2(vors, vor_update_3, disp_vors, divs, div_update_3, disp_divs, height, height_update_3, disp_heights, area, disp_areas, 0.5*run_config.delta_t));

	// // bve_forcing(run_config, xcos, ycos, zcos, disp_vors, effective_vorticity, time);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, disp_areas, disp_vors, proxy_source_vors);
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, disp_divs, proxy_source_divs);
	// upward_pass(run_config, interp_vals, cubed_sphere_panels, disp_areas, disp_heights, proxy_source_heights);
	swe_vel_interactions_2(run_config, disp_x, disp_y, disp_z, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, proxy_source_divs, interaction_list, cubed_sphere_panels, interp_vals);
	// laplacian_interactions(run_config, disp_x, disp_y, disp_z, proxy_target_h, proxy_source_heights, interaction_list, cubed_sphere_panels, interp_vals);
	downward_pass_3(run_config, interp_vals, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x_4, vel_y_4, vel_z_4);
	// downward_pass(run_config, interp_vals, cubed_sphere_panels, proxy_target_h, height_lap_4);

	MPI_Allreduce(MPI_IN_PLACE, &vel_x_4(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y_4(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z_4(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// MPI_Allreduce(MPI_IN_PLACE, &height_lap_4(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_x_4, 1e-9));
	// Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_y_4, 1e-9));
	Kokkos::parallel_for(run_config.active_panel_count, filter_vals(vel_z_4, 1e-8));
	// Kokkos::parallel_for(run_config.active_panel_count, swe_height_correction(xcos, ycos, zcos, vel_x_3, vel_y_3, vel_z_3, height, height_update_3, height_lap_4, cubed_sphere_panels, offset, run_config.delta_t, run_config.interp_degree));

	Kokkos::View<double**, Kokkos::LayoutRight> vor_update_4 ("vorticity k4", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> div_update_4 ("divergence k4", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> height_update_4 ("height k4", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(run_config.active_panel_count, swe_tendency_computation_2(xcos, ycos, zcos, vel_x_4, vel_y_4, vel_z_4, vors, vor_update_4, divs, div_update_4, height, height_update_4, cubed_sphere_panels, run_config.omega,run_config.interp_degree, offset));

	// compute vor/div/h at arrival points
	Kokkos::View<double**, Kokkos::LayoutRight> arrival_vor ("arrival vor", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> new_divs ("new divs", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> new_h ("new heights", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(run_config.active_panel_count, swe_compute_arrival_values_2(xcos, ycos, zcos, vors, divs, height, arrival_vor, new_divs, new_h, vel_x, vel_y, vel_z, vel_x_2, vel_y_2, vel_z_2, vel_x_3, vel_y_3, vel_z_3, 
																				vel_x_4, vel_y_4, vel_z_4, vor_update_1, div_update_1, height_update_1, vor_update_2, div_update_2, height_update_2, vor_update_3, div_update_3, height_update_3, 
																				vor_update_4, div_update_4, height_update_4, cubed_sphere_panels, run_config.delta_t, run_config.interp_degree, offset));

	// backwards RK4 to find departure points
	Kokkos::View<double**, Kokkos::LayoutRight> dep_x ("departure points x", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> dep_y ("departure points y", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> dep_z ("departure points z", run_config.active_panel_count, dim2size);
	Kokkos::parallel_for(xcos.extent_int(0), swe_find_departure_points_2(xcos, ycos, zcos, vel_x, vel_y, vel_z, vel_x_2, vel_y_2, vel_z_2, vel_x_3, vel_y_3, vel_z_3, vel_x_4, vel_y_4, vel_z_4, dep_x, dep_y, dep_z, cubed_sphere_panels, run_config.delta_t, offset, run_config.interp_degree));
	
	// compute update from departure points to target points
	Kokkos::View<double**, Kokkos::LayoutRight> new_vors ("new vorticities", run_config.active_panel_count, dim2size);
	Kokkos::View<double***, Kokkos::LayoutRight> new_passive_tracers("new passive tarcers", run_config.active_panel_count, dim2size, run_config.tracer_count);
	
	Kokkos::parallel_for(run_config.active_panel_count, swe_departure_to_target_2(dep_x, dep_y, dep_z, arrival_vor, new_vors, passive_tracers, new_passive_tracers, cubed_sphere_panels, run_config.interp_degree, offset, run_config.omega));

	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {vors.extent_int(0), vors.extent_int(1)}), copy_kokkos_view_2(vors, new_vors));
	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {divs.extent_int(0), divs.extent_int(1)}), copy_kokkos_view_2(divs, new_divs));
	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {height.extent_int(0), height.extent_int(1)}), copy_kokkos_view_2(height, new_h));
	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0, 0}, {passive_tracers.extent_int(0), passive_tracers.extent_int(1), passive_tracers.extent_int(2)}), copy_kokkos_view_3(passive_tracers, new_passive_tracers));
}
