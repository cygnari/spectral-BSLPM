#ifndef H_SWE_TIME_STEP_LL_H
#define H_SWE_TIME_STEP_LL_H

#include <Kokkos_Core.hpp>
#include "run_config.hpp"

void swe_back_rk4_step_ll(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& area, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vors, Kokkos::View<double**, Kokkos::LayoutRight>& divs, Kokkos::View<double**, Kokkos::LayoutRight>& height, 
							Kokkos::View<double**, Kokkos::LayoutRight>& topo, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, double time);

#endif
