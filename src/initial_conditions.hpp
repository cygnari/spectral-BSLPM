#ifndef H_INITIAL_CONDITIONS_H
#define H_INITIAL_CONDITIONS_H

#include "Kokkos_Core.hpp"
#include "run_config.hpp"

void poisson_initialize(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& xcos, 
						Kokkos::View<double*, Kokkos::HostSpace>& ycos, 
						Kokkos::View<double*, Kokkos::HostSpace>& zcos, 
						Kokkos::View<double*, Kokkos::HostSpace>& potential);

void poisson_initialize(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos,
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& potential);

void bve_initialize(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area);

void tracer_initialize(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity, 
						Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace>& passive_tracers);

#endif