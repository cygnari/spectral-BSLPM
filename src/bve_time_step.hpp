#ifndef H_BVE_TIME_STEP_H
#define H_BVE_TIME_STEP_H

#include <Kokkos_Core.hpp>
#include "run_config.hpp"

void bve_back_rk4_step(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& ycos, Kokkos::View<double**, Kokkos::LayoutRight>& zcos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& area, Kokkos::View<double**, Kokkos::LayoutRight>& vors, 
							Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_vors, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
							Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, Kokkos::View<double***, Kokkos::LayoutRight>& passive_tracers, double time);

#endif
