#ifndef H_FMM_H
#define H_FMM_H

#include <Kokkos_Core.hpp>

#include "initialize_cubed_sphere.hpp"
#include "run_config.hpp"

struct interact_pair {
	int target_panel;
	int source_panel;
	int interact_type; // 0 = pp, 1 = pc, 2 = cp, 3 = cc
};

void dual_tree_traversal(RunConfig& run_config, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels, 
							Kokkos::View<interact_pair*, Kokkos::HostSpace>& interaction_list);

void upward_pass(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& area,
					Kokkos::View<double**, Kokkos::LayoutRight>& pots, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_pots);

void downward_pass(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
					Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots, 
					Kokkos::View<double**, Kokkos::LayoutRight>& sols);

#endif
