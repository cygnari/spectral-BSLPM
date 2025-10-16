#ifndef H_DIRECT_SUM_IMPL_H
#define H_DIRECT_SUM_IMPL_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"

struct inv_lap {
	Kokkos::View<double*> xcos;
	Kokkos::View<double*> ycos;
	Kokkos::View<double*> zcos;
	Kokkos::View<double*> area;
	Kokkos::View<double*> pots;
	Kokkos::View<double*> soln;

	inv_lap(Kokkos::View<double*>& xcos_, Kokkos::View<double*>& ycos_, Kokkos::View<double*>& zcos_, 
			Kokkos::View<double*>& area_, Kokkos::View<double*>& pots_, Kokkos::View<double*>& soln_) :
			xcos(xcos_), ycos(ycos_), zcos(zcos_), area(area_), pots(pots_), soln(soln_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		if (i != j) {
			double dp = xcos(i) * xcos(j) + ycos(i) * ycos(j) + zcos(i) * zcos(j);
			// constant is -1/(4pi)
			Kokkos::atomic_add(&soln(i), -0.07957747154594767 * Kokkos::log(1-dp) * area(j) * pots(j));
		}
	}
};

#endif
