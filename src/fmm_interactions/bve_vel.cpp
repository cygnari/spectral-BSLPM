#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "bve_vel_impl.hpp"

void bve_vel_interactions(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& disp_x, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& disp_z, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& source_vals, Kokkos::View<interact_pair*>& interaction_list, 
                                Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals) {
    // first calculate subset of interactions to compute
    int ints[run_config.mpi_p], lbs[run_config.mpi_p], ubs[run_config.mpi_p];
    for (int i = 0; i < run_config.mpi_p; i++) {
        ints[i] = int(run_config.fmm_interaction_count / run_config.mpi_p);
    } 
    int total = run_config.mpi_p * ints[0];
    int gap = run_config.fmm_interaction_count - total;
    for (int i = 1; i < gap + 1; i++) {
        ints[i] += 1;
    }
    lbs[0] = 0;
    ubs[0] = ints[0];
    for (int i = 1; i < run_config.mpi_p; i++) {
        lbs[i] = ubs[i-1];
        ubs[i] = lbs[i] + ints[i];
    }
    int lb, ub;
    lb = lbs[run_config.mpi_id];
    ub = ubs[run_config.mpi_id];

    int offset = cubed_sphere_panels.extent_int(0) - run_config.active_panel_count;
    Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), bve_vel_panel_interaction(disp_x, disp_y, disp_z, target_pots_1, target_pots_2, target_pots_3, source_vals, interp_vals, cubed_sphere_panels, interaction_list, offset, run_config.kernel_eps));
    Kokkos::fence();
}
