#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "general_utils_impl.hpp"

struct poisson_panel_interaction {
	Kokkos::View<double**, Kokkos::LayoutRight> target_pots;
	Kokkos::View<double**, Kokkos::LayoutRight> source_vals;
	Kokkos::View<double**, Kokkos::LayoutRight> interp_vals;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	Kokkos::View<interact_pair*> interaction_list;

	poisson_panel_interaction(Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_,
								Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals_, Kokkos::View<CubedSpherePanel*> cubed_sphere_panels_, 
								Kokkos::View<interact_pair*>& interaction_list_) : target_pots(target_pots_), source_vals(source_vals_), 
								interp_vals(interp_vals_), cubed_sphere_panels(cubed_sphere_panels_), interaction_list(interaction_list_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_panel = interaction_list(i).target_panel;
		int source_panel = interaction_list(i).source_panel;
		double xi_off_t, xi_scale_t, xi_off_s, xi_scale_s, eta_off_t, eta_scale_t, eta_off_s, eta_scale_s;
		xi_off_t = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(target_panel).max_xi + cubed_sphere_panels(target_panel).min_xi);
		xi_scale_t = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(target_panel).max_xi - cubed_sphere_panels(target_panel).min_xi);
		xi_off_s = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(source_panel).max_xi + cubed_sphere_panels(source_panel).min_xi);
		xi_scale_s = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(source_panel).max_xi - cubed_sphere_panels(source_panel).min_xi);

		eta_off_t = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(target_panel).max_eta + cubed_sphere_panels(target_panel).min_eta);
		eta_scale_t = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(target_panel).max_eta - cubed_sphere_panels(target_panel).min_eta);
		eta_off_s = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(source_panel).max_eta + cubed_sphere_panels(source_panel).min_eta);
		eta_scale_s = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(source_panel).max_eta - cubed_sphere_panels(source_panel).min_eta);

		double xi_s, eta_s, xi_t, eta_t, xyz_t[3], xyz_s[3], dp;
		for (int j = 0; j < interp_vals.extent_int(0); j++) { // target loop
			xi_t = interp_vals(j,0) * xi_scale_t + xi_off_t;
			eta_t = interp_vals(j,1) * eta_scale_t + eta_off_t;
			xyz_from_xieta(xi_t, eta_t, cubed_sphere_panels(target_panel).face, xyz_t);
			for (int k = 0; k < interp_vals.extent(0); k++) { // source loop
				xi_s = interp_vals(k,0) * xi_scale_s + xi_off_s;
				eta_s = interp_vals(k,1) * eta_scale_s + eta_off_s;
				xyz_from_xieta(xi_s, eta_s, cubed_sphere_panels(source_panel).face, xyz_s);
				dp = xyz_t[0] * xyz_s[0] + xyz_t[1] * xyz_s[1] + xyz_t[2] * xyz_s[2];
				if (dp < 1.0 - 1e-15) { // constant is -1/(4pi)
					Kokkos::atomic_add(&target_pots(target_panel, j), -0.07957747154594767 * Kokkos::log(1-dp) * source_vals(source_panel, k));
				} 
				// else {
					// Kokkos::atomic_add(&target_pots(target_panel, j), source_vals(source_panel,k));
				// }
			}
		}
	}
};

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
		lbs[i] = ubs[i];
		ubs[i] = lbs[i] + ints[i];
	}
	int lb, ub;
	lb = lbs[run_config.mpi_id];
	ub = ubs[run_config.mpi_id];

	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), poisson_panel_interaction(target_pots, source_vals, interp_vals, cubed_sphere_panels, interaction_list));
	Kokkos::fence();
}