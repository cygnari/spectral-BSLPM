#ifndef H_FMM_POISSON_H
#define H_FMM_POISSON_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"

void poisson_fmm_interactions(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots, 
								Kokkos::View<double**, Kokkos::LayoutRight>& source_vals, Kokkos::View<interact_pair*>& interaction_list, 
								Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals);

#endif