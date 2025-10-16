#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "inverse_poisson_impl.hpp"

void poisson_fmm_interactions(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots, 
								Kokkos::View<double**, Kokkos::LayoutRight>& source_vals, Kokkos::View<interact_pair*>& interaction_list, 
								Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals) {
	// first compute list of interactions to compute
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

	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), poisson_panel_interaction(target_pots, source_vals, interp_vals, cubed_sphere_panels, interaction_list));
	Kokkos::fence();
}