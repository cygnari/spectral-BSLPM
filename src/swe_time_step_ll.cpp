#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "fmm_funcs.hpp"
#include "fmm_interactions/swe_vel.hpp"
#include "forcing_funcs.hpp"
#include "general_utils_impl.hpp"

void swe_back_rk4_step_ll(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& area, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vors, Kokkos::View<double**, Kokkos::LayoutRight>& divs, Kokkos::View<double**, Kokkos::LayoutRight>& height, 
							Kokkos::View<double**, Kokkos::LayoutRight>& topo, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, double time) {
	// SWE single rk4 step lat lon coordinates
	int dim2size = (run_config.interp_degree+1)*(run_config.interp_degree+1);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_vors ("proxy source vors", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_divs ("proxy source divs", run_config.panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_divs.extent_int(0), proxy_source_divs.extent_int(1)}), zero_out(proxy_source_divs));
	upward_pass_ll(run_config, xcos, ycos, zcos, cubed_sphere_panels, area, vors, proxy_source_vors, leaf_panel_points);
	upward_pass_ll(run_config, xcos, ycos, zcos, cubed_sphere_panels, area, divs, proxy_source_divs, leaf_panel_points);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_1("proxy target vels", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_2("proxy target vels", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_3("proxy target vels", run_config.panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_1.extent_int(0), proxy_target_1.extent_int(1)}), zero_out(proxy_target_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_2.extent_int(0), proxy_target_2.extent_int(1)}), zero_out(proxy_target_2));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_3.extent_int(0), proxy_target_3.extent_int(1)}), zero_out(proxy_target_3));
	swe_vel_interactions_ll(run_config, xcos, ycos, zcos, proxy_target_1, proxy_target_2, proxy_target_3, proxy_source_vors, proxy_source_divs, vors, divs, area, interaction_list, cubed_sphere_panels, leaf_panel_points);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_x.extent_int(0), vel_x.extent_int(1)}), zero_out(vel_x));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_y.extent_int(0), vel_y.extent_int(1)}), zero_out(vel_y));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_z.extent_int(0), vel_z.extent_int(1)}), zero_out(vel_z));
	downward_pass_3_ll(run_config, xcos, ycos, zcos, cubed_sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x, vel_y, vel_z, leaf_panel_points);
	MPI_Allreduce(MPI_IN_PLACE, &vel_x(0,0), run_config.lat_count*run_config.lon_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y(0,0), run_config.lat_count*run_config.lon_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z(0,0), run_config.lat_count*run_config.lon_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}