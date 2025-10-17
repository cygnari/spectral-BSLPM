#ifndef H_FMM_IMPL_H
#define H_FMM_IMPL_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "cubed_sphere_transforms.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "general_utils.hpp"
#include "interp_funcs.hpp"
#include "interp_funcs_impl.hpp"

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
		double* bli_vals = (double*) Kokkos::kokkos_malloc(sizeof(double) * point_count);
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
		Kokkos::kokkos_free(bli_vals);
	}
};

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
		// double bli_vals[point_count];
		double* bli_vals = (double*) Kokkos::kokkos_malloc(sizeof(double) * point_count);
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
		Kokkos::kokkos_free(bli_vals);
	}
};

struct parent_to_child_3 {
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots_1;
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots_2;
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots_3;
	Kokkos::View<double**, Kokkos::LayoutRight> interp_vals;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int point_count;
	int interp_degree;

	parent_to_child_3(Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_1_, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_2_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_3_, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals_,
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, int point_count_, int interp_degree_) : proxy_target_pots_1(proxy_target_pots_1_),
					proxy_target_pots_2(proxy_target_pots_2_), proxy_target_pots_3(proxy_target_pots_3_), 
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
		// double bli_vals[point_count];
		double* bli_vals = (double*) Kokkos::kokkos_malloc(sizeof(double) * point_count);
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
				Kokkos::atomic_add(&proxy_target_pots_1(child,k), bli_vals[l]*proxy_target_pots_1(i,l));
				Kokkos::atomic_add(&proxy_target_pots_2(child,k), bli_vals[l]*proxy_target_pots_2(i,l));
				Kokkos::atomic_add(&proxy_target_pots_3(child,k), bli_vals[l]*proxy_target_pots_3(i,l));
			}
		}
		Kokkos::kokkos_free(bli_vals);
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

#endif
