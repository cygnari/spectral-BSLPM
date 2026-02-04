#ifndef H_TOPO_H
#define H_TOPO_H

#include <Kokkos_Core.hpp>
#include "run_config.hpp"

void apply_topography(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, Kokkos::View<double**, Kokkos::LayoutRight>& zcos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& disp_x, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y, Kokkos::View<double**, Kokkos::LayoutRight>& disp_z, 
							Kokkos::View<double**, Kokkos::LayoutRight>& height, Kokkos::View<double**, Kokkos::LayoutRight>& effective_height);

void apply_topography_2(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& disp_x, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& disp_y, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& disp_z, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& height, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& effective_height);

#endif
