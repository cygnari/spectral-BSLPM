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
#include <iostream>

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
		// double* bli_vals = (double*) Kokkos::kokkos_malloc(sizeof(double) * point_count);
		double bli_vals[121];
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
		// Kokkos::kokkos_free(bli_vals);
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
		// double* bli_vals = (double*) Kokkos::kokkos_malloc(sizeof(double) * point_count);
		double bli_vals[121];
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
		// Kokkos::kokkos_free(bli_vals);
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
		double bli_vals[121];
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

struct base_pots_ll {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> area;
	Kokkos::View<double**, Kokkos::LayoutRight> pots;
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_pots;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	Kokkos::View<int**, Kokkos::LayoutRight> leaf_panel_points;
	int degree;

	base_pots_ll(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& area_, Kokkos::View<double**, Kokkos::LayoutRight>& pots_, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_pots_, 
				Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points_, int degree_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), 
				area(area_), pots(pots_), proxy_source_pots(proxy_source_pots_), cubed_sphere_panels(cubed_sphere_panels_), leaf_panel_points(leaf_panel_points_), degree(degree_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		// panel index i
		double xieta[2], basis_vals[121];
		int index, index_i, index_j;
		int lon_count = xcos.extent_int(1);
		if (cubed_sphere_panels(i).is_leaf) {
			for (int j = 0; j < cubed_sphere_panels(i).point_count; j++) {
				index = leaf_panel_points(i,j);
				index_j = index % lon_count;
				index_i = index / lon_count;
				xieta_from_xyz(xcos(index_i, index_j), ycos(index_i, index_j), zcos(index_i, index_j), xieta);
				interp_vals_bli(basis_vals, xieta[0], xieta[1], cubed_sphere_panels(i).min_xi, cubed_sphere_panels(i).max_xi, cubed_sphere_panels(i).min_eta, cubed_sphere_panels(i).max_eta, degree);
				for (int k = 0; k < (degree+1)*(degree+1); k++) {
					proxy_source_pots(i,k) += basis_vals[k] * pots(index_i, index_j) * area(index_i, index_j);
				}
			}
		}
	}
};

struct child_to_parent_ll {
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_pots;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int degree;

	child_to_parent_ll(Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_pots_, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, int degree_) :
						proxy_source_pots(proxy_source_pots_), cubed_sphere_panels(cubed_sphere_panels_), degree(degree_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const { // upward pass
		int parent = cubed_sphere_panels(i).parent_id;
		double min_xi, max_xi, min_eta, max_eta, xi, eta, xi_off, xi_scale, eta_off, eta_scale;
		double bli_vals[121];
		int index, index2;
		min_xi = cubed_sphere_panels(parent).min_xi;
		max_xi = cubed_sphere_panels(parent).max_xi;
		min_eta = cubed_sphere_panels(parent).min_eta;
		max_eta = cubed_sphere_panels(parent).max_eta;
		xi_off = 0.5*(cubed_sphere_panels(i).min_xi + cubed_sphere_panels(i).max_xi);
		xi_scale = 0.5*(cubed_sphere_panels(i).max_xi - cubed_sphere_panels(i).min_xi);
		eta_off = 0.5*(cubed_sphere_panels(i).min_eta + cubed_sphere_panels(i).max_eta);
		eta_scale = 0.5*(cubed_sphere_panels(i).max_eta - cubed_sphere_panels(i).min_eta);
		for (int j = 0; j < degree+1; j++) { // xi index
			for (int k = 0; k < degree+1; k++) { // eta index
				xi = Kokkos::cos(Kokkos::numbers::pi/degree*j)*xi_scale + xi_off;
				eta = Kokkos::cos(Kokkos::numbers::pi/degree*k)*eta_scale + eta_off;
				interp_vals_bli(bli_vals, xi, eta, min_xi, max_xi, min_eta, max_eta, degree);
				index = j*(degree+1)+k;
				for (int l = 0; l < degree+1; l++) {
					for (int m = 0; m < degree+1; m++) {
						index2 = l*(degree+1)+m;
						proxy_source_pots(parent,index2) += bli_vals[index2]*proxy_source_pots(i,index);
					}
				}
			}
		}
	}
};

struct parent_to_child_3_ll {
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots_1;
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots_2;
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots_3;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int interp_degree;

	parent_to_child_3_ll(Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_1_, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_2_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_3_, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, int interp_degree_) : 
					proxy_target_pots_1(proxy_target_pots_1_), proxy_target_pots_2(proxy_target_pots_2_), proxy_target_pots_3(proxy_target_pots_3_), 
					cubed_sphere_panels(cubed_sphere_panels_), interp_degree(interp_degree_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const { // downward pass
		if (not cubed_sphere_panels(i).is_leaf) {
			int child;
			int k1, k2;
			if (j == 0) {
				child = cubed_sphere_panels(i).child1;
			} else if (j == 1) {
				child = cubed_sphere_panels(i).child2;
			} else if (j == 2) {
				child = cubed_sphere_panels(i).child3;
			} else {
				child = cubed_sphere_panels(i).child4;
			}
			// std::cout << "parent: " << i << " child: " << child << " " << j << std::endl;
			double min_xi, max_xi, min_eta, max_eta, xi, eta, xi_off, xi_scale, eta_off, eta_scale;
			double bli_vals[121];
			min_xi = cubed_sphere_panels(i).min_xi;
			max_xi = cubed_sphere_panels(i).max_xi;
			min_eta = cubed_sphere_panels(i).min_eta;
			max_eta = cubed_sphere_panels(i).max_eta;
			xi_off = 0.5*(cubed_sphere_panels(child).min_xi + cubed_sphere_panels(child).max_xi);
			xi_scale = 0.5*(cubed_sphere_panels(child).max_xi - cubed_sphere_panels(child).min_xi);
			eta_off = 0.5*(cubed_sphere_panels(child).min_eta + cubed_sphere_panels(child).max_eta);
			eta_scale = 0.5*(cubed_sphere_panels(child).max_eta - cubed_sphere_panels(child).min_eta);
			// std::cout << min_xi << " " << max_xi << " " << min_eta << " " << max_eta << std::endl;
			// std::cout << cubed_sphere_panels(child).min_xi << " " << cubed_sphere_panels(child).max_xi << " " << cubed_sphere_panels(child).min_eta << " " << cubed_sphere_panels(child).max_eta << std::endl;

			for (int k = 0; k < (interp_degree+1)*(interp_degree+1); k++) { // loop over points in child panel
				k1 = k / interp_degree;
				k2 = k % interp_degree;
				// std::cout << k << " " << k1 << " " << k2 << std::endl;
				xi = Kokkos::cos(Kokkos::numbers::pi/interp_degree*k1)*xi_scale + xi_off;
				eta = Kokkos::cos(Kokkos::numbers::pi/interp_degree*k2)*eta_scale + eta_off;
				// std::cout << xi << " " << min_xi << " " << max_xi << std::endl;
				// std::cout << eta << " " << min_eta << " " << max_eta << std::endl;
				interp_vals_bli(bli_vals, xi, eta, min_xi, max_xi, min_eta, max_eta, interp_degree);
				for (int l = 0; l < (interp_degree+1)*(interp_degree+1); l++) {
					Kokkos::atomic_add(&proxy_target_pots_1(child,k), bli_vals[l]*proxy_target_pots_1(i,l));
					Kokkos::atomic_add(&proxy_target_pots_2(child,k), bli_vals[l]*proxy_target_pots_2(i,l));
					Kokkos::atomic_add(&proxy_target_pots_3(child,k), bli_vals[l]*proxy_target_pots_3(i,l));
				}
			}
		}
	}
};

struct child_panel_interp_3_ll {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots_1;
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots_2;
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_pots_3;
	Kokkos::View<double**, Kokkos::LayoutRight> target_pots_1;
	Kokkos::View<double**, Kokkos::LayoutRight> target_pots_2;
	Kokkos::View<double**, Kokkos::LayoutRight> target_pots_3;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	Kokkos::View<int**, Kokkos::LayoutRight> leaf_panel_points;
	int degree;

	child_panel_interp_3_ll(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
							Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_1_, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_2_, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_target_pots_3_,
							Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3_, 
							Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points_, int degree_) : 
							xcos(xcos_), ycos(ycos_), zcos(zcos_), proxy_target_pots_1(proxy_target_pots_1_), proxy_target_pots_2(proxy_target_pots_2_), proxy_target_pots_3(proxy_target_pots_3_), 
							target_pots_1(target_pots_1_), target_pots_2(target_pots_2_), target_pots_3(target_pots_3_), 
							cubed_sphere_panels(cubed_sphere_panels_), leaf_panel_points(leaf_panel_points_), degree(degree_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const { // interp from leaf panel to target points contained inside
		if (cubed_sphere_panels(i).is_leaf) {
			double xieta[2], bli_vals[121], min_xi, max_xi, min_eta, max_eta;
			int index, index_i, index_j;
			int lon_count = xcos.extent_int(1);
			min_xi = cubed_sphere_panels(i).min_xi;
			max_xi = cubed_sphere_panels(i).max_xi;
			min_eta = cubed_sphere_panels(i).min_eta;
			max_eta = cubed_sphere_panels(i).max_eta;
			for (int j = 0; j < cubed_sphere_panels(i).point_count; j++) {
				index = leaf_panel_points(i,j);
				index_j = index % lon_count;
				index_i = index / lon_count;
				xieta_from_xyz(xcos(index_i, index_j), ycos(index_i, index_j), zcos(index_i, index_j), xieta);
				interp_vals_bli(bli_vals, xieta[0], xieta[1], min_xi, max_xi, min_eta, max_eta, degree);
				for (int k = 0; k < (degree+1)*(degree+1); k++) {
					target_pots_1(index_i,index_j) += bli_vals[k]*proxy_target_pots_1(i,k);
					target_pots_2(index_i,index_j) += bli_vals[k]*proxy_target_pots_2(i,k);
					target_pots_3(index_i,index_j) += bli_vals[k]*proxy_target_pots_3(i,k);
				}
			}
		}
	}
};

#endif
