#include <cmath>
#include <string>

#include "Kokkos_Core.hpp"
#include "run_config.hpp"

struct sh43 {
	Kokkos::View<double*, Kokkos::HostSpace> xcos;
	Kokkos::View<double*, Kokkos::HostSpace> ycos;
	Kokkos::View<double*, Kokkos::HostSpace> zcos;
	Kokkos::View<double*, Kokkos::HostSpace> potential;

	sh43(Kokkos::View<double*, Kokkos::HostSpace> xcos_, Kokkos::View<double*, Kokkos::HostSpace> ycos_, Kokkos::View<double*, Kokkos::HostSpace> zcos_, Kokkos::View<double*, Kokkos::HostSpace> potential_) :
		xcos(xcos_), ycos(ycos_), zcos(zcos_), potential(potential_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double x = xcos(i);
		double y = ycos(i);
		potential(i) = 1.770130769779931*x*(x*x-3.0*y*y)*zcos(i); // constant is 3.0/4.0*sqrt(35.0/(2.0*M_PI))
	}
};

void poisson_initialize(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& xcos, Kokkos::View<double*, Kokkos::HostSpace>& ycos, Kokkos::View<double*, Kokkos::HostSpace>& zcos, Kokkos::View<double*, Kokkos::HostSpace>& potential) {
	if (run_config.initial_condition == "sh43") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.point_count), sh43(xcos, ycos, zcos, potential));
	}
	Kokkos::fence();
}