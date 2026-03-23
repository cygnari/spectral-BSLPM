#ifndef H_FMM_SWE_H
#define H_FMM_SWE_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "initialize_octo_sphere.hpp"

void swe_vel_interactions(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& disp_x, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& disp_z, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_vor, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_div, 
                                Kokkos::View<interact_pair*>& interaction_list,  Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals);

void swe_vel_interactions_2(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& disp_x, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& disp_z, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_vor, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_div, 
                                Kokkos::View<interact_pair*>& interaction_list,  Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals);

void swe_vel_interactions_ll(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_vor, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_div, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& vors, Kokkos::View<double**, Kokkos::LayoutRight>& divs, Kokkos::View<double**, Kokkos::LayoutRight>& area,
                                Kokkos::View<interact_pair*>& interaction_list,  Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points);

void swe_vel_interactions_octo(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_vor, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_div, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& vors, Kokkos::View<double**, Kokkos::LayoutRight>& divs, Kokkos::View<double**, Kokkos::LayoutRight>& area,
                                Kokkos::View<interact_pair*>& interaction_list,  Kokkos::View<OctoSpherePanel*> octo_sphere_panels, Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points);

#endif
