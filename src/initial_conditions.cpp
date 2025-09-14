#include <cmath>
#include <string>

#include "Kokkos_Core.hpp"
#include "run_config.hpp"

struct sh43 {
	Kokkos::View<double*> xcos;
	Kokkos::View<double*> ycos;
	Kokkos::View<double*> zcos;
	Kokkos::View<double*> potential;

	sh43(Kokkos::View<double*> xcos_, Kokkos::View<double*> ycos_, Kokkos::View<double*> zcos_, Kokkos::View<double*> potential_) :
		xcos(xcos_), ycos(ycos_), zcos(zcos_), potential(potential_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double x = xcos(i);
		double y = ycos(i);
		potential(i) = 1.770130769779931*x*(x*x-3.0*y*y)*zcos(i); // constant is 3.0/4.0*sqrt(35.0/(2.0*M_PI))
	}
};

void poisson_initialize(const RunConfig& run_config, Kokkos::View<double*>& xcos, Kokkos::View<double*>& ycos, Kokkos::View<double*>& zcos, Kokkos::View<double*>& potential) {
	if (run_config.initial_condition == "sh43") {
		Kokkos::parallel_for(run_config.point_count, sh43(xcos, ycos, zcos, potential));
	}
	Kokkos::fence();
}