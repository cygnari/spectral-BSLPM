#include <cmath>
#include <Kokkos_Core.hpp>
#include <iostream>

#include "cubed_sphere_transforms_impl.hpp"
#include "general_utils_impl.hpp"
#include "initialize_cubed_sphere.hpp"
#include "run_config.hpp"
#include "deriv_funcs_impl.hpp"

void xyz_gradient(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutStride>& x_comps, 
					Kokkos::View<double**, Kokkos::LayoutStride>& y_comps, Kokkos::View<double**, Kokkos::LayoutStride>& z_comps, 
					Kokkos::View<double**, Kokkos::LayoutRight>& func_vals, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels) {
	int ub = run_config.panel_count;
	int lb = run_config.panel_count - run_config.active_panel_count;
	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), panel_gradient(x_comps, y_comps, z_comps, func_vals, cubed_sphere_panels, run_config.interp_degree, lb));
}

void xyz_gradient_xieta(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutStride>& xi_comps, 
					Kokkos::View<double**, Kokkos::LayoutStride>& eta_comps, Kokkos::View<double**, Kokkos::LayoutRight>& func_vals, 
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels) {
	int ub = run_config.panel_count;
	int lb = run_config.panel_count - run_config.active_panel_count;
	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), panel_gradient_xieta(xi_comps, eta_comps, func_vals, cubed_sphere_panels, run_config.interp_degree, lb));
}

void laplacian(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& laplacian_vals, 
					Kokkos::View<double**, Kokkos::LayoutRight>& func_vals, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels) {
	int ub = run_config.panel_count;
	int lb = run_config.panel_count - run_config.active_panel_count;
	Kokkos::View<double**, Kokkos::LayoutRight> xi_derivs_workspace ("xi derivs workspace", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
	Kokkos::View<double**, Kokkos::LayoutRight> eta_derivs_workspace ("eta derivs workspace", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
	Kokkos::View<double**, Kokkos::LayoutRight> xieta_derivs_workspace ("xieta derivs workspace", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
	Kokkos::View<double**, Kokkos::LayoutRight> xixi_derivs_workspace ("xixi derivs workspace", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
	Kokkos::View<double**, Kokkos::LayoutRight> etaeta_derivs_workspace ("etaeta derivs workspace", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), panel_laplacian(laplacian_vals, func_vals, xi_derivs_workspace, eta_derivs_workspace, xieta_derivs_workspace, 
																		xixi_derivs_workspace, etaeta_derivs_workspace, cubed_sphere_panels, run_config.interp_degree, lb));
}