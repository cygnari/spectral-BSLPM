#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "fmm_funcs.hpp"
#include "fmm_interactions/bve_vel.hpp"
#include "newton_solve.hpp"

#include "cubed_sphere_transforms_impl.hpp"
#include "interp_funcs_impl.hpp"

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
			xi = xieta[0];
			eta = xieta[1];
			panel_index = point_locate_panel(cubed_sphere_panels, dep_x(i,j), dep_y(i,j), dep_z(i,j));
			interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], cubed_sphere_panels(panel_index).min_xi, cubed_sphere_panels(panel_index).max_xi, 
							cubed_sphere_panels(panel_index).min_eta, cubed_sphere_panels(panel_index).max_eta, degree);
			new_vors(i,j) = 0;
			for (int k = 0; k < passive_tracers.extent_int(2); k++) {
				new_passive_tracers(i,j,k) = 0;
			}
			vor = 0;
			for (int m = 0; m < degree+1; m++) { // xi loop
				for (int n = 0; n < degree+1; n++) { // eta loop
					one_d_index = m * (degree + 1) + n;
					vor += vors(panel_index - offset, one_d_index) * bli_basis_vals[one_d_index];
					for (int k = 0; k < passive_tracers.extent_int(2); k++) {
						new_passive_tracers(i,j,k) += passive_tracers(panel_index - offset, one_d_index, k) * bli_basis_vals[one_d_index];
					}
				}
			}	
			vor += 2 * omega * dep_z(i,j);
			new_vors(i,j) = vor - 2*omega*zcos(i,j);
		}
	}
};

void bve_back_newton_step(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& ycos, Kokkos::View<double**, Kokkos::LayoutRight>& zcos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& area, Kokkos::View<double**, Kokkos::LayoutRight>& vors, 
							Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_vors, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
							Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, Kokkos::View<double***, Kokkos::LayoutRight>& passive_tracers) {
	// first do upward pass
	upward_pass(run_config, interp_vals, cubed_sphere_panels, area, vors, proxy_source_vors);
	int dim2size = (run_config.interp_degree+1)*(run_config.interp_degree+1);

	// next do velocity computation
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_1("proxy target vels 1", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_2("proxy target vels 1", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_3("proxy target vels 1", run_config.panel_count, dim2size);
	bve_vel_interactions(run_config, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, interaction_list, cubed_sphere_panels, interp_vals);

	// downward pass
	downward_pass_3(run_config, interp_vals, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x, vel_y, vel_z);

	// newton solve to find departure points
	Kokkos::View<double**, Kokkos::LayoutRight> dep_x ("departure points x", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> dep_y ("departure points y", run_config.active_panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> dep_z ("departure points z", run_config.active_panel_count, dim2size);
	newton_solve(run_config, xcos, ycos, zcos, vel_x, vel_y, vel_z, dep_x, dep_y, dep_z, cubed_sphere_panels);

	// compute update from departure points to target points
	Kokkos::View<double**, Kokkos::LayoutRight> new_vors ("new vorticities", run_config.active_panel_count, dim2size);
	Kokkos::View<double***, Kokkos::LayoutRight> new_passive_tracers("new passive tarcers", run_config.active_panel_count, dim2size, run_config.tracer_count);
	int offset = cubed_sphere_panels.extent_int(0) - run_config.active_panel_count;
	Kokkos::parallel_for(run_config.active_panel_count, departure_to_target(
		zcos, dep_x, dep_y, dep_z, vors, new_vors, passive_tracers, new_passive_tracers, cubed_sphere_panels, run_config.interp_degree, offset, run_config.omega));

	vors = new_vors;
	passive_tracers = new_passive_tracers;
}
