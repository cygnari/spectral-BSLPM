#ifndef H_DERIV_FUNCS_H
#define H_DERIV_FUNCS_H

#include <Kokkos_Core.hpp>

#include "initialize_cubed_sphere.hpp"
#include "run_config.hpp"

void bli_deriv_xi(double* deriv_vals, double* func_vals, int degree, double min_xi, double max_xi);

void bli_deriv_eta(double* deriv_vals, double* func_vals, int degree, double min_eta, double max_eta);

void xyz_gradient(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutStride>& x_comps, 
					Kokkos::View<double**, Kokkos::LayoutStride>& y_comps, Kokkos::View<double**, Kokkos::LayoutStride>& z_comps, 
					Kokkos::View<double**, Kokkos::LayoutRight>& func_vals, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels);

void xyz_gradient_xieta(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutStride>& xi_comps, 
					Kokkos::View<double**, Kokkos::LayoutStride>& eta_comps, Kokkos::View<double**, Kokkos::LayoutRight>& func_vals, 
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels);

void laplacian(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& laplacian_vals, 
					Kokkos::View<double**, Kokkos::LayoutRight>& func_vals, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels);

// void single_panel_grad(double* x_comps, double* y_comps, double* z_comps, double* f_vals, int degree, double min_xi, double max_xi, double min_eta, double max_eta, int face);

#endif
