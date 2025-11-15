#include <Kokkos_Core.hpp>
#include <queue>
#include <iostream>
#include <vector>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "cubed_sphere_transforms.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "general_utils.hpp"
#include "interp_funcs.hpp"
#include "interp_funcs_impl.hpp"
#include "fmm_funcs_impl.hpp"

void dual_tree_traversal(RunConfig& run_config, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels, 
							Kokkos::View<interact_pair*, Kokkos::HostSpace>& interaction_list) {
	std::vector<interact_pair> temp_interaction_list;
	int interaction_count = 0;

	std::queue<int> target_squares;
	std::queue<int> source_squares;
	// top level interactions
	int ub = 6;
	int lb = 0;
	for (int i = lb; i < ub; i++) {
		for (int j = lb; j < ub; j++) {
			target_squares.push(i);
			source_squares.push(j);
		}
	}

	int index_target, index_source;
	double c1[3], c2[3], dist, separation, xi1, xi2, eta1, eta2;

	while (target_squares.size() > 0) {
		index_target = target_squares.front();
		index_source = source_squares.front();
		
		target_squares.pop();
		source_squares.pop();
		xi1 = M_PI/4.0*0.5*(cubed_sphere_panels(index_target).min_xi + cubed_sphere_panels(index_target).max_xi);
		xi2 = M_PI/4.0*0.5*(cubed_sphere_panels(index_source).min_xi + cubed_sphere_panels(index_source).max_xi);
		eta1 = M_PI/4.0*0.5*(cubed_sphere_panels(index_target).min_eta + cubed_sphere_panels(index_target).max_eta);
		eta2 = M_PI/4.0*0.5*(cubed_sphere_panels(index_source).min_eta + cubed_sphere_panels(index_source).max_eta);
		xyz_from_xieta(xi1, eta1, cubed_sphere_panels(index_target).face, c1);
		xyz_from_xieta(xi2, eta2, cubed_sphere_panels(index_source).face, c2);
		dist = gcdist(c1, c2);
		separation = 100;
		if (dist > 1e-16) {
			separation = (cubed_sphere_panels(index_target).radius + cubed_sphere_panels(index_source).radius) / dist;
		}
		if (separation < run_config.fmm_theta) {
			// well separated
			interact_pair new_interact = {index_target, index_source, 0};
			if (cubed_sphere_panels(index_target).level < run_config.levels - 1) {
				new_interact.interact_type += 2;
			} 
			if (cubed_sphere_panels(index_source).level < run_config.levels - 1) {
				new_interact.interact_type += 1;
			}
			temp_interaction_list.push_back(new_interact);
			interaction_count += 1;
		} else {
			// not well separated, split up panel at higher level, preferentially split target
			if (cubed_sphere_panels(index_target).is_leaf and cubed_sphere_panels(index_source).is_leaf) {
				// both are leaf panels
				interact_pair new_interact {index_target, index_source, 0};
				temp_interaction_list.push_back(new_interact);
				interaction_count += 1;
			} else if (cubed_sphere_panels(index_target).is_leaf) {
				// target is leaf, break up source
				target_squares.push(index_target);
				target_squares.push(index_target);
				target_squares.push(index_target);
				target_squares.push(index_target);
				source_squares.push(cubed_sphere_panels(index_source).child1);
				source_squares.push(cubed_sphere_panels(index_source).child2);
				source_squares.push(cubed_sphere_panels(index_source).child3);
				source_squares.push(cubed_sphere_panels(index_source).child4);
			} else if (cubed_sphere_panels(index_source).is_leaf) {
				source_squares.push(index_source);
				source_squares.push(index_source);
				source_squares.push(index_source);
				source_squares.push(index_source);
				target_squares.push(cubed_sphere_panels(index_target).child1);
				target_squares.push(cubed_sphere_panels(index_target).child2);
				target_squares.push(cubed_sphere_panels(index_target).child3);
				target_squares.push(cubed_sphere_panels(index_target).child4);
			} else {
				// neither is leaf
				if (cubed_sphere_panels(index_target).level <= cubed_sphere_panels(index_source).level) {
					// refine target
					source_squares.push(index_source);
					source_squares.push(index_source);
					source_squares.push(index_source);
					source_squares.push(index_source);
					target_squares.push(cubed_sphere_panels(index_target).child1);
					target_squares.push(cubed_sphere_panels(index_target).child2);
					target_squares.push(cubed_sphere_panels(index_target).child3);
					target_squares.push(cubed_sphere_panels(index_target).child4);
				} else {
					// refine source
					target_squares.push(index_target);
					target_squares.push(index_target);
					target_squares.push(index_target);
					target_squares.push(index_target);
					source_squares.push(cubed_sphere_panels(index_source).child1);
					source_squares.push(cubed_sphere_panels(index_source).child2);
					source_squares.push(cubed_sphere_panels(index_source).child3);
					source_squares.push(cubed_sphere_panels(index_source).child4);
				}
			}
		}
	}	

	resize(interaction_list, interaction_count);
	for (int i = 0; i < interaction_count; i++) {
		interaction_list(i) = temp_interaction_list[i];
	}
	run_config.fmm_interaction_count = interaction_count;
}

void upward_pass(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& area,
					Kokkos::View<double**, Kokkos::LayoutRight>& pots, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_pots) {

	int lb, ub, pc;
	ub = cubed_sphere_panels.extent_int(0);
	lb = ub - run_config.active_panel_count;
	pc = pow(run_config.interp_degree+1, 2);
	Kokkos::parallel_for(Kokkos::RangePolicy(0, run_config.active_panel_count), base_pots(area, pots, proxy_source_pots, pow(run_config.interp_degree+1, 2), lb));
	for (int i = run_config.levels; i > 1; i--) {		
		Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), child_to_parent(proxy_source_pots, interp_vals, cubed_sphere_panels, pc, run_config.interp_degree));
		ub = lb;
		lb = ub - 6 * pow(4, i-2);
	}
}

void downward_pass(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
					Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots, 
					Kokkos::View<double**, Kokkos::LayoutRight>& sols) {
	int lb, ub, pc;
	lb = 0;
	ub = 0;
	pc = pow(run_config.interp_degree+1, 2);
	for (int i = 0; i < run_config.levels - 1; i++) {
		lb = ub;
		ub = lb + 6 * pow(4, i);
		Kokkos::parallel_for(Kokkos::MDRangePolicy({lb, 0}, {ub, 4}), parent_to_child(proxy_target_pots, interp_vals, cubed_sphere_panels, pc, run_config.interp_degree));
	}

	Kokkos::parallel_for(run_config.active_panel_count, child_panel_copy(proxy_target_pots, sols, ub));
	Kokkos::fence();
}

void downward_pass_3(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
					Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_1, 
					Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_2, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_3,  
					Kokkos::View<double**, Kokkos::LayoutRight>& sols_1, Kokkos::View<double**, Kokkos::LayoutRight>& sols_2, Kokkos::View<double**, Kokkos::LayoutRight>& sols_3) {
	int lb, ub, pc;
	lb = 0;
	ub = 0;
	pc = pow(run_config.interp_degree+1, 2);
	for (int i = 0; i < run_config.levels - 1; i++) {
		lb = ub;
		ub = lb + 6 * pow(4, i);
		Kokkos::parallel_for(Kokkos::MDRangePolicy({lb, 0}, {ub, 4}), parent_to_child_3(proxy_target_pots_1, proxy_target_pots_2, proxy_target_pots_3, interp_vals, cubed_sphere_panels, pc, run_config.interp_degree));
	}

	Kokkos::parallel_for(run_config.active_panel_count, child_panel_copy(proxy_target_pots_1, sols_1, ub));
	Kokkos::parallel_for(run_config.active_panel_count, child_panel_copy(proxy_target_pots_2, sols_2, ub));
	Kokkos::parallel_for(run_config.active_panel_count, child_panel_copy(proxy_target_pots_3, sols_3, ub));
	Kokkos::fence();
}




