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

void swe_initialize(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& divergence, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& height, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& topo);

void swe_initialize_ll(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& lats, Kokkos::View<double*, Kokkos::HostSpace>& lons, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vor, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& div, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& height, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& topo);

#endif