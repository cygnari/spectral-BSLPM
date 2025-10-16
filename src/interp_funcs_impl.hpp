#ifndef H_INTERP_IMPL_H
#define H_INTERP_IMPL_H

#include "run_config.hpp"

KOKKOS_INLINE_FUNCTION 
void interp_vals_bli(double* basis_vals, double xi, double eta, double min_xi, double max_xi, 
						double min_eta, double max_eta, int interp_deg) {
	double cheb_xi[interp_deg+1], cheb_eta[interp_deg+1], bli_weights[interp_deg+1], xi_range, xi_offset, eta_range, eta_offset, val;
	xi_range = 0.5*(max_xi - min_xi);
	xi_offset = 0.5*(max_xi + min_xi);
	eta_range = 0.5*(max_eta - min_eta);
	eta_offset = 0.5*(max_eta + min_eta);

	if (interp_deg == 0) {
		cheb_xi[0] = xi_offset;
		cheb_eta[0] = eta_offset;
		bli_weights[0] = 1;
	} else {
		for (int i = 0; i < interp_deg+1; i++) {
			cheb_xi[i] = Kokkos::cos(Kokkos::numbers::pi * i / interp_deg) * xi_range + xi_offset;
			cheb_eta[i] = Kokkos::cos(Kokkos::numbers::pi * i / interp_deg) * eta_range + eta_offset;
			if (i == 0) {
				bli_weights[i] = 0.5;
			} else if (i == interp_deg) {
				bli_weights[i] = 0.5 * Kokkos::pow(-1, i);
			} else {
				bli_weights[i] = pow(-1, i);
			}
		}
	}

	bool found_xi_point = false;
	double xi_f_vals[interp_deg+1], denom_xi;
	for (int i = 0; i < interp_deg+1; i++) {
		if (Kokkos::abs(xi - cheb_xi[i]) < 1e-16) {
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
			val = bli_weights[i] / (xi - cheb_xi[i]);
			xi_f_vals[i] = val;
			denom_xi += val;
		}
		for (int i = 0; i < interp_deg+1; i++) {
			xi_f_vals[i] /= denom_xi;
		}
	}

	bool found_eta_point = false;
	double eta_f_vals[interp_deg+1], denom_eta;
	for (int i = 0; i < interp_deg+1; i++) {
		if (Kokkos::abs(eta - cheb_eta[i]) < 1e-16) {
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
			val = bli_weights[i] / (eta - cheb_eta[i]);
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

#endif
