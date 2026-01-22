#ifndef H_FORCING_FUNCS_H
#define H_FORCING_FUNCS_H

#include "Kokkos_Core.hpp"
#include "run_config.hpp"

void bve_forcing(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, 
						Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
						Kokkos::View<double**, Kokkos::LayoutRight>& zcos, 
						Kokkos::View<double**, Kokkos::LayoutRight>& vorticity, 
						Kokkos::View<double**, Kokkos::LayoutRight>& effective_vorticity, double time);

#endif