#ifndef H_INTERP_IMPL_H
#define H_INTERP_IMPL_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "initialize_cubed_sphere.hpp"
#include "cubed_sphere_transforms_impl.hpp"

KOKKOS_INLINE_FUNCTION 
void interp_vals_bli(double* basis_vals, double xi, double eta, double min_xi, double max_xi, 
						double min_eta, double max_eta, int interp_deg) {
	double xi_range, xi_offset, eta_range, eta_offset, val, chebxi, chebeta;
	xi_range = 0.5*(max_xi - min_xi);
	xi_offset = 0.5*(max_xi + min_xi);
	eta_range = 0.5*(max_eta - min_eta);
	eta_offset = 0.5*(max_eta + min_eta);
	double xi_f_vals[121], eta_f_vals[121];

	bool found_xi_point = false;
	double denom_xi, bliweight;
	for (int i = 0; i < interp_deg+1; i++) {
		chebxi = Kokkos::cos(Kokkos::numbers::pi * i / interp_deg) * xi_range + xi_offset;
		if (Kokkos::abs(xi - chebxi) < 1e-16) {
			found_xi_point = true;
			for (int j = 0; j < interp_deg+1; j++) {
				xi_f_vals[j] = 0;
			}
			xi_f_vals[i] = 1;
			break;
		}
	}
	if (not found_xi_point) {
		denom_xi = 0;
		for (int i = 0; i < interp_deg+1; i++) {
			chebxi = Kokkos::cos(Kokkos::numbers::pi * i / interp_deg) * xi_range + xi_offset;
			if (i == 0) {
				bliweight = 0.5;
			} else if (i == interp_deg) {
				bliweight = 0.5 * Kokkos::pow(-1, i);
			} else {
				bliweight = pow(-i, i);
			}
			val = bliweight / (xi - chebxi);
			xi_f_vals[i] = val;
			denom_xi += val;
		}
		for (int i = 0; i < interp_deg+1; i++) {
			xi_f_vals[i] /= denom_xi;
		}
	}

	bool found_eta_point = false;
	double denom_eta;
	for (int i = 0; i < interp_deg+1; i++) {
		chebeta = Kokkos::cos(Kokkos::numbers::pi * i / interp_deg) * eta_range + eta_offset;
		if (Kokkos::abs(eta - chebeta) < 1e-16) {
			found_eta_point = true;
			for (int j = 0; j < interp_deg+1; j++) {
				eta_f_vals[j] = 0;
			}
			eta_f_vals[i] = 1;
			break;
		}
	}
	if (not found_eta_point) {
		denom_eta = 0;
		for (int i = 0; i < interp_deg+1; i++) {
			chebeta = Kokkos::cos(Kokkos::numbers::pi * i / interp_deg) * eta_range + eta_offset;
			if (i == 0) {
				bliweight = 0.5;
			} else if (i == interp_deg) {
				bliweight = 0.5 * Kokkos::pow(-1, i);
			} else {
				bliweight = pow(-i, i);
			}
			val = bliweight / (eta - chebeta);
			eta_f_vals[i] = val;
			denom_eta += val;
		}
		for (int i = 0; i < interp_deg+1; i++) {
			eta_f_vals[i] /= denom_eta;
		}
	}

	int index;
	for (int i = 0; i < interp_deg+1; i++) {
		for (int j = 0; j < interp_deg+1; j++) {
			index = i * (interp_deg+1) + j;
			basis_vals[index] = xi_f_vals[i] * eta_f_vals[j];
		}
	}
}

KOKKOS_INLINE_FUNCTION
int point_locate_panel(Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, double x, double y, double z) {
	// finds the leaf panel that contains the point (x,y,z)
	int curr_index;
	double xi, eta, xieta[2], mid_eta, mid_xi;

	curr_index = face_from_xyz(x,y,z);

	xieta_from_xyz(x, y, z, curr_index, xieta);
	// xi = xieta[0] * 1.27323954473516269; // constant is 4/pi
	// eta = xieta[1] * 1.27323954473516269;
	xi = xieta[0];
	eta = xieta[1];

	while (not cubed_sphere_panels(curr_index).is_leaf) {
		// iterate through until leaf, compare xi eta
		mid_xi = 0.5*(cubed_sphere_panels(curr_index).max_xi + cubed_sphere_panels(curr_index).min_xi);
		mid_eta = 0.5*(cubed_sphere_panels(curr_index).max_eta + cubed_sphere_panels(curr_index).min_eta);
		if ((xi >= mid_xi) and (eta >= mid_eta)) {
			curr_index = cubed_sphere_panels(curr_index).child1;
		} else if ((xi <= mid_xi) and (eta >= mid_eta)) {
			curr_index = cubed_sphere_panels(curr_index).child2;
		} else if ((xi <= mid_xi) and (eta <= mid_eta)) {
			curr_index = cubed_sphere_panels(curr_index).child3;
		} else if ((xi >= mid_xi) and (eta <= mid_eta)) {
			curr_index = cubed_sphere_panels(curr_index).child4;
		} else {
			Kokkos::abort("Error in point locate panel");
		}
	}
	return curr_index;
}

// KOKKOS_INLINE_FUNCTION
// double interp_eval(Kokkos::View<double*>& func_vals, double xi, double eta, double min_xi, double max_xi,
// 					double min_eta, double max_eta, int interp_deg) {
// 	double xi_range, xi_offset, eta_range, eta_offset;
// 	xi_range = 0.5*(max_xi - min_xi);
// 	xi_offset = 0.5*(max_xi + min_xi);
// 	eta_range = 0.5*(max_eta - min_eta);
// 	eta_offset = 0.5*(max_eta + min_eta);
// 	double xi_comp, eta_comp;

	
// }

#endif
