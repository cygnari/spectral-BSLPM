#ifndef H_INTERP_H
#define H_INTERP_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "initialize_cubed_sphere.hpp"

void interp_vals_bli(double* basis_vals, double xi, double eta, double min_xi, double max_xi, 
						double min_eta, double max_eta, int interp_deg);

void interp_init(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_info);

int point_locate_panel(Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, double x, double y, double z);

// double interp_eval(Kokkos::View<double*>& func_vals, double xi, double eta, double min_xi, double max_xi,
// 					double min_eta, double max_eta, int interp_deg);

void interp_to_latlon(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& data, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels, 
											Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interped_data, Kokkos::View<double*, Kokkos::HostSpace>& lats, Kokkos::View<double*, Kokkos::HostSpace>& lons);

#endif
