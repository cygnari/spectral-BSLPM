#ifndef H_FMM_BVE_H
#define H_FMM_BVE_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"

void bve_vel_interactions(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& source_vals, Kokkos::View<interact_pair*>& interaction_list, 
                                Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals);
#endif
