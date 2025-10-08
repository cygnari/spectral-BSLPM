#ifndef H_DERIV_FUNCS_IMPL_H
#define H_DERIV_FUNCS_IMPL_H

#include <Kokkos_Core.hpp>

inline void bli_deriv_xi(double* deriv_vals, double* func_vals, int degree, double min_xi, double max_xi) {
	// computes the xi derivative using barycentric Lagrange differentiation
	double cheb_xi[degree+1], bli_weights[degree+1], xi_range, xi_offset;
	xi_range = 0.5*(max_xi - min_xi);
	xi_offset = 0.5*(max_xi + min_xi);

	for (int i = 0; i < degree+1; i++) {
		cheb_xi[i] = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_range + xi_offset;
		if (i == 0) {
			bli_weights[i] = 0.5;
		} else if (i == degree) {
			bli_weights[i] = 0.5 * Kokkos::pow(-1, i);
		} else {
			bli_weights[i] = Kokkos::pow(-1, i);
		}
	}

	int index, index2;
	double fi, wi, xi;
	for (int j = 0; j < degree+1; j++) {
		// fixed eta value
		for (int i = 0; i < degree+1; j++) {
			index = j * (degree+1) + i;
			deriv_vals[index] = 0;
			fi = func_vals[index];
			wi = bli_weights[i];
			xi = cheb_xi[i];
			for (int k = 0; k < degree+1; k++) {
				if (i != k) {
					index2 = j * (degree+1) + k;
					deriv_vals[index] += (func_vals[index2]-fi) * bli_weights[k] / (xi - cheb_xi[k]);
				}
			}
			deriv_vals[index] /= wi;
		}
	}
}

inline void bli_deriv_eta(double* deriv_vals, double* func_vals, int degree, double min_eta, double max_eta) {
	// computes the xi derivative using barycentric Lagrange differentiation
	double cheb_eta[degree+1], bli_weights[degree+1], eta_range, eta_offset;
	eta_range = 0.5*(max_eta - min_eta);
	eta_offset = 0.5*(max_eta + min_eta);

	for (int i = 0; i < degree+1; i++) {
		cheb_eta[i] = Kokkos::cos(Kokkos::numbers::pi * i / degree) * eta_range + eta_offset;
		if (i == 0) {
			bli_weights[i] = 0.5;
		} else if (i == degree) {
			bli_weights[i] = 0.5 * Kokkos::pow(-1, i);
		} else {
			bli_weights[i] = Kokkos::pow(-1, i);
		}
	}

	int index, index2;
	double fi, wi, yi;
	for (int i = 0; i < degree+1; i++) {
		// fixed xi value
		for (int j = 0; j < degree+1; j++) {
			index = j * (degree+1) + i;
			deriv_vals[index] = 0;
			fi = func_vals[index];
			wi = bli_weights[i];
			yi = cheb_eta[i];
			for (int k = 0; k < degree+1; k++) {
				if (j != k) {
					index2 = j * (degree+1) + k;
					deriv_vals[index] += (func_vals[index2]-fi) * bli_weights[k] / (yi - cheb_eta[k]);
				}
			}
			deriv_vals[index] /= wi;
		}
	}
}

#endif
