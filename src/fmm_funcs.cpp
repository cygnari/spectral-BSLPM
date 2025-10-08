#include <Kokkos_Core.hpp>
#include <queue>
#include <iostream>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "cubed_sphere_transforms.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "general_utils.hpp"
#include "interp_funcs.hpp"
#include "interp_funcs_impl.hpp"

void dual_tree_traversal(RunConfig& run_config, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels, 
							Kokkos::View<interact_pair*, Kokkos::HostSpace>& interaction_list) {
	Kokkos::View<interact_pair*, Kokkos::HostSpace> temp_interaction_list ("temp interaction list", pow(run_config.active_panel_count, 2));
	int interaction_count = 0;

	std::queue<int> target_squares;
	std::queue<int> source_squares;
	// top level interactions
	int ub = 6;
	int lb = 0;
	// int ub = run_config.panel_count;
	// int lb = run_config.panel_count - run_config.active_panel_count;
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
			temp_interaction_list(interaction_count) = new_interact;
			interaction_count += 1;
		} else {
			// not well separated, split up panel at higher level, preferentially split target
			if (cubed_sphere_panels(index_target).is_leaf and cubed_sphere_panels(index_source).is_leaf) {
				// both are leaf panels
				interact_pair new_interact {index_target, index_source, 0};
				temp_interaction_list(interaction_count) = new_interact;
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

	resize(temp_interaction_list, interaction_count);
	resize(interaction_list, interaction_count);
	Kokkos::deep_copy(interaction_list, temp_interaction_list);
	run_config.fmm_interaction_count = interaction_count;
}

struct base_pots {
	Kokkos::View<double**, Kokkos::LayoutRight> area;
	Kokkos::View<double**, Kokkos::LayoutRight> pots;
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_pots;
	int point_count;
	int offset;

	base_pots(Kokkos::View<double**, Kokkos::LayoutRight>& area_, Kokkos::View<double**, Kokkos::LayoutRight>& pots_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_pots_, int point_count_, int offset_) : area(area_), pots(pots_), 
				proxy_source_pots(proxy_source_pots_), point_count(point_count_), offset(offset_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		for (int j = 0; j < point_count; j++) {
			proxy_source_pots(offset+i,j) = area(i,j) * pots(i,j);
		}
	}
};

struct child_to_parent {
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_pots;
	Kokkos::View<double**, Kokkos::LayoutRight> interp_vals;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int point_count;
	int interp_degree;

	child_to_parent(Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_pots_, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals_,
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, int point_count_, int interp_degree_) : proxy_source_pots(proxy_source_pots_), 
					interp_vals(interp_vals_), cubed_sphere_panels(cubed_sphere_panels_), point_count(point_count_), interp_degree(interp_degree_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int k) const { // k is child panel index
		int parent = cubed_sphere_panels(k).parent_id;
		double min_xi, max_xi, min_eta, max_eta, xi, eta, xi_off, xi_scale, eta_off, eta_scale;
		double bli_vals[point_count];
		min_xi = cubed_sphere_panels(parent).min_xi * Kokkos::numbers::pi / 4.0;
		max_xi = cubed_sphere_panels(parent).max_xi * Kokkos::numbers::pi / 4.0;
		min_eta = cubed_sphere_panels(parent).min_eta * Kokkos::numbers::pi / 4.0;
		max_eta = cubed_sphere_panels(parent).max_eta * Kokkos::numbers::pi / 4.0;
		xi_off = Kokkos::numbers::pi/4.0*0.5*(cubed_sphere_panels(k).min_xi + cubed_sphere_panels(k).max_xi);
		xi_scale = Kokkos::numbers::pi/4.0*0.5*(cubed_sphere_panels(k).max_xi - cubed_sphere_panels(k).min_xi);
		eta_off = Kokkos::numbers::pi/4.0*0.5*(cubed_sphere_panels(k).min_eta + cubed_sphere_panels(k).max_eta);
		eta_scale = Kokkos::numbers::pi/4.0*0.5*(cubed_sphere_panels(k).max_eta - cubed_sphere_panels(k).min_eta);
		for (int i = 0; i < point_count; i++) {
			xi = interp_vals(i,0) * xi_scale + xi_off;
			eta = interp_vals(i,1) * eta_scale + eta_off;
			interp_vals_bli(bli_vals, xi, eta, min_xi, max_xi, min_eta, max_eta, interp_degree);
			for (int j = 0; j < point_count; j++) {
				Kokkos::atomic_add(&proxy_source_pots(parent,j), bli_vals[j] * proxy_source_pots(k,i));
			}
		}
	}
};

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

struct parent_to_child {
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots;
	Kokkos::View<double**, Kokkos::LayoutRight> interp_vals;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int point_count;
	int interp_degree;

	parent_to_child(Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals_,
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, int point_count_, int interp_degree_) : proxy_target_pots(proxy_target_pots_),
					interp_vals(interp_vals_), cubed_sphere_panels(cubed_sphere_panels_), point_count(point_count_), interp_degree(interp_degree_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		int child;
		if (j == 0) {
			child = cubed_sphere_panels(i).child1;
		} else if (j == 1) {
			child = cubed_sphere_panels(i).child2;
		} else if (j == 2) {
			child = cubed_sphere_panels(i).child3;
		} else {
			child = cubed_sphere_panels(i).child4;
		}
		double min_xi, max_xi, min_eta, max_eta, xi, eta, xi_off, xi_scale, eta_off, eta_scale;
		double bli_vals[point_count];
		double val;
		min_xi = cubed_sphere_panels(i).min_xi * Kokkos::numbers::pi / 4.0;
		max_xi = cubed_sphere_panels(i).max_xi * Kokkos::numbers::pi / 4.0;
		min_eta = cubed_sphere_panels(i).min_eta * Kokkos::numbers::pi / 4.0;
		max_eta = cubed_sphere_panels(i).max_eta * Kokkos::numbers::pi / 4.0;
		xi_off = Kokkos::numbers::pi/4.0*0.5*(cubed_sphere_panels(child).min_xi + cubed_sphere_panels(child).max_xi);
		xi_scale = Kokkos::numbers::pi/4.0*0.5*(cubed_sphere_panels(child).max_xi - cubed_sphere_panels(child).min_xi);
		eta_off = Kokkos::numbers::pi/4.0*0.5*(cubed_sphere_panels(child).min_eta + cubed_sphere_panels(child).max_eta);
		eta_scale = Kokkos::numbers::pi/4.0*0.5*(cubed_sphere_panels(child).max_eta - cubed_sphere_panels(child).min_eta);

		for (int k = 0; k < point_count; k++) { // loop over points in child panel
			xi = interp_vals(k,0)*xi_scale + xi_off;
			eta = interp_vals(k,1)*eta_scale + eta_off;
			interp_vals_bli(bli_vals, xi, eta, min_xi, max_xi, min_eta, max_eta, interp_degree);
			for (int l = 0; l < point_count; l++) {
				val = bli_vals[l] * proxy_target_pots(i,l);

				Kokkos::atomic_add(&proxy_target_pots(child,k), bli_vals[l]*proxy_target_pots(i,l));
			}
		}
	}
};

struct child_panel_copy {
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots;
	Kokkos::View<double**, Kokkos::LayoutRight> sols;
	int offset;

	child_panel_copy(Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_, Kokkos::View<double**, Kokkos::LayoutRight>& sols_, 
					 int offset_) : proxy_target_pots(proxy_target_pots_), sols(sols_), offset(offset_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		for (int j = 0; j < sols.extent_int(1); j++) {
			sols(i,j) = proxy_target_pots(i+offset,j);
		}
	}
};

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




