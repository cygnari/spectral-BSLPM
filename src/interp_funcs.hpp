#ifndef H_INTERP_H
#define H_INTERP_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"

void interp_vals_bli(double* basis_vals, double xi, double eta, double min_xi, double max_xi, 
						double min_eta, double max_eta, int interp_deg);

void interp_init(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_info);

#endif
