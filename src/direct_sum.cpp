#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "run_config.hpp"
#include "mpi_utils.hpp"

struct inv_lap {
	Kokkos::View<double*> xcos;
	Kokkos::View<double*> ycos;
	Kokkos::View<double*> zcos;
	Kokkos::View<double*> area;
	Kokkos::View<double*> pots;
	Kokkos::View<double*> soln;

	inv_lap(Kokkos::View<double*> xcos_, Kokkos::View<double*> ycos_, Kokkos::View<double*> zcos_, Kokkos::View<double*> area_, Kokkos::View<double*> pots_, Kokkos::View<double*> soln_) :
			xcos(xcos_), ycos(ycos_), zcos(zcos_), area(area_), pots(pots_), soln(soln_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		if (i != j) {
			double dp = xcos(i) * xcos(j) + ycos(i) * ycos(j) + zcos(i) * zcos(j);
			Kokkos::atomic_add(&soln(i), -0.07957747154594767 * Kokkos::log(1-dp) * area(j) * pots(j));
		}
	}
};

void direct_sum_inv_lap(const RunConfig& run_config, Kokkos::View<double*> xcos, Kokkos::View<double*> ycos, Kokkos::View<double*> zcos, Kokkos::View<double*> area, Kokkos::View<double*> pots, Kokkos::View<double*> soln) {
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {run_config.oned_lb, 0},{run_config.oned_ub, run_config.point_count}), inv_lap(xcos, ycos, zcos, area, pots, soln));
	Kokkos::fence();

	MPI_Allreduce(MPI_IN_PLACE, &soln(0), run_config.point_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
}