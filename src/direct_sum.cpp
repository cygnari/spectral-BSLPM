#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "run_config.hpp"
#include "mpi_utils.hpp"
#include "direct_sum_impl.hpp"

void direct_sum_inv_lap(const RunConfig& run_config, Kokkos::View<double*> xcos, 
						Kokkos::View<double*> ycos, Kokkos::View<double*> zcos, 
						Kokkos::View<double*> area, Kokkos::View<double*> pots, 
						Kokkos::View<double*> soln) {
	// Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left,Kokkos::Iterate::Left>>(Kokkos::DefaultExecutionSpace(), {run_config.oned_lb, 0},{run_config.oned_ub, run_config.point_count}), inv_lap(xcos, ycos, zcos, area, pots, soln));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {run_config.oned_lb, 0},{run_config.oned_ub, run_config.point_count}), inv_lap(xcos, ycos, zcos, area, pots, soln));
	Kokkos::fence();

	MPI_Allreduce(MPI_IN_PLACE, &soln(0), run_config.point_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
}