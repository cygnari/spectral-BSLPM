#include <cmath>
#include <Kokkos_Core.hpp>
#include <iostream>

#include "cubed_sphere_transforms_impl.hpp"
#include "general_utils_impl.hpp"
#include "initialize_cubed_sphere.hpp"
#include "run_config.hpp"
#include "deriv_funcs_impl.hpp"

// void bli_deriv_xi(double* deriv_vals, double* func_vals, int degree, double min_xi, double max_xi) {
// 	// computes the xi derivative using barycentric Lagrange differentiation
// 	double cheb_xi[degree+1], bli_weights[degree+1], xi_range, xi_offset;
// 	xi_range = 0.5*(max_xi - min_xi);
// 	xi_offset = 0.5*(max_xi + min_xi);

// 	for (int i = 0; i < degree+1; i++) {
// 		cheb_xi[i] = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_range + xi_offset;
// 		if (i == 0) {
// 			bli_weights[i] = 0.5;
// 		} else if (i == degree) {
// 			bli_weights[i] = 0.5 * Kokkos::pow(-1, i);
// 		} else {
// 			bli_weights[i] = Kokkos::pow(-1, i);
// 		}
// 	}

// 	int index, index2;
// 	double fi, wi, xi;
// 	for (int j = 0; j < degree+1; j++) {
// 		// fixed eta value
// 		for (int i = 0; i < degree+1; i++) {
// 			index = i * (degree+1) + j;
// 			deriv_vals[index] = 0;
// 			fi = func_vals[index];
// 			wi = bli_weights[i];
// 			xi = cheb_xi[i];
// 			for (int k = 0; k < degree+1; k++) {
// 				if (i != k) {
// 					index2 = k * (degree+1) + j;
// 					deriv_vals[index] += (func_vals[index2]-fi) * bli_weights[k] / (xi - cheb_xi[k]);
// 				}
// 			}
// 			deriv_vals[index] /= wi;
// 		}
// 	}
// }

// void bli_deriv_eta(double* deriv_vals, double* func_vals, int degree, double min_eta, double max_eta) {
// 	// computes the xi derivative using barycentric Lagrange differentiation
// 	double cheb_eta[degree+1], bli_weights[degree+1], eta_range, eta_offset;
// 	eta_range = 0.5*(max_eta - min_eta);
// 	eta_offset = 0.5*(max_eta + min_eta);

// 	for (int i = 0; i < degree+1; i++) {
// 		cheb_eta[i] = Kokkos::cos(Kokkos::numbers::pi * i / degree) * eta_range + eta_offset;
// 		if (i == 0) {
// 			bli_weights[i] = 0.5;
// 		} else if (i == degree) {
// 			bli_weights[i] = 0.5 * Kokkos::pow(-1, i);
// 		} else {
// 			bli_weights[i] = Kokkos::pow(-1, i);
// 		}
// 	}

// 	int index, index2;
// 	double fi, wi, yi;
// 	for (int i = 0; i < degree+1; i++) {
// 		// fixed xi value
// 		for (int j = 0; j < degree+1; j++) {
// 			index = i * (degree+1) + j;
// 			deriv_vals[index] = 0;
// 			fi = func_vals[index];
// 			wi = bli_weights[j];
// 			yi = cheb_eta[j];
// 			for (int k = 0; k < degree+1; k++) {
// 				if (j != k) {
// 					index2 = i * (degree+1) + k;
// 					deriv_vals[index] += (func_vals[index2]-fi) * bli_weights[k] / (yi - cheb_eta[k]);
// 				}
// 			}
// 			deriv_vals[index] /= wi;
// 		}
// 	}
// }

void xyz_gradient(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& x_comps, 
					Kokkos::View<double**, Kokkos::LayoutRight>& y_comps, Kokkos::View<double**, Kokkos::LayoutRight>& z_comps, 
					Kokkos::View<double**, Kokkos::LayoutRight>& func_vals, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels) {
	int ub = run_config.panel_count;
	int lb = run_config.panel_count - run_config.active_panel_count;
	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), panel_gradient(x_comps, y_comps, z_comps, func_vals, cubed_sphere_panels, run_config.interp_degree, lb));
}

void laplacian(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& laplacian_vals, 
					Kokkos::View<double**, Kokkos::LayoutRight>& func_vals, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels) {
	int ub = run_config.panel_count;
	int lb = run_config.panel_count - run_config.active_panel_count;
	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), panel_laplacian(laplacian_vals, func_vals, cubed_sphere_panels, run_config.interp_degree, lb));
}