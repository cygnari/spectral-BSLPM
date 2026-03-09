#ifndef H_SWE_TIME_STEP_2_H
#define H_SWE_TIME_STEP_2_H

#include <Kokkos_Core.hpp>
#include "run_config.hpp"

void swe_back_rk4_step_2(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& area, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vors, Kokkos::View<double**, Kokkos::LayoutRight>& divs, Kokkos::View<double**, 
							Kokkos::LayoutRight>& height, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals, 
							Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, Kokkos::View<double***, Kokkos::LayoutRight>& passive_tracers, Kokkos::View<double**, Kokkos::LayoutRight>& base_vor_tend, 
							Kokkos::View<double**, Kokkos::LayoutRight>& base_div_tend, Kokkos::View<double**, Kokkos::LayoutRight>& base_h_tend, Kokkos::View<int*>& one_d_no_of_points, Kokkos::View<int**, Kokkos::LayoutRight>& two_d_to_1d, double time);

#endif
