#include <cmath>
#include <string>
#include <iostream>

#include <Kokkos_Core.hpp>
#include "run_config.hpp"
#include "general_utils.hpp"

struct sh43 {
	Kokkos::View<double*, Kokkos::HostSpace> xcos;
	Kokkos::View<double*, Kokkos::HostSpace> ycos;
	Kokkos::View<double*, Kokkos::HostSpace> zcos;
	Kokkos::View<double*, Kokkos::HostSpace> potential;

	sh43(Kokkos::View<double*, Kokkos::HostSpace>& xcos_, Kokkos::View<double*, Kokkos::HostSpace> ycos_, 
			Kokkos::View<double*, Kokkos::HostSpace>& zcos_, Kokkos::View<double*, Kokkos::HostSpace>& potential_) :
		xcos(xcos_), ycos(ycos_), zcos(zcos_), potential(potential_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double x = xcos(i);
		double y = ycos(i);
		potential(i) = 1.770130769779931*x*(x*x-3.0*y*y)*zcos(i); // constant is 3.0/4.0*sqrt(35.0/(2.0*M_PI))
	}
};

void poisson_initialize(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& xcos, 
						Kokkos::View<double*, Kokkos::HostSpace>& ycos, 
						Kokkos::View<double*, Kokkos::HostSpace>& zcos, 
						Kokkos::View<double*, Kokkos::HostSpace>& potential) {
	if (run_config.initial_condition == "sh43") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.point_count), sh43(xcos, ycos, zcos, potential));
	}
	Kokkos::fence();
}

struct ones_2{
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> potential;

	ones_2(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& potential_) : potential(potential_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = potential.extent_int(1);
		for (int j = 0; j < jmax; j++) {
			potential(i,j) = 1.0;
		}
	}
};

struct sh43_2{
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> potential;

	sh43_2(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& potential_) :
		xcos(xcos_), ycos(ycos_), zcos(zcos_), potential(potential_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double x, y;
		for (int j = 0; j < jmax; j++) {
			x = xcos(i, j);
			y = ycos(i, j);
			potential(i, j) = 1.770130769779931*x*(x*x-3.0*y*y)*zcos(i, j); // constant is 3.0/4.0*sqrt(35.0/(2.0*M_PI))
		}
	}
};

struct coslon_2{
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> potential;

	coslon_2(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& potential_) :
		xcos(xcos_), ycos(ycos_), zcos(zcos_), potential(potential_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double theta, x, y;
		for (int j = 0; j < jmax; j++) {
			x = xcos(i, j);
			y = ycos(i, j);
			theta = Kokkos::atan2(y,x);
			potential(i,j) = Kokkos::cos(theta);
		}
	}
};

struct sinlat_2{
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> potential;

	sinlat_2(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& potential_) :
		xcos(xcos_), ycos(ycos_), zcos(zcos_), potential(potential_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double theta, x, y;
		for (int j = 0; j < jmax; j++) {
			x = xcos(i, j);
			y = ycos(i, j);
			theta = Kokkos::atan2(Kokkos::sqrt(x*x+y*y),zcos(i,j));
			potential(i,j) = Kokkos::sin(Kokkos::numbers::pi/2.0-theta);
		}
	}
};

void poisson_initialize(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos,
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& potential) {
	if (run_config.initial_condition == "sh43") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), sh43_2(xcos, ycos, zcos, potential));
	} else if (run_config.initial_condition == "ones") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), ones_2(potential));
	} else if (run_config.initial_condition == "coslon") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), coslon_2(xcos, ycos, zcos, potential));
	} else if (run_config.initial_condition == "sinlat") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), sinlat_2(xcos, ycos, zcos, potential));
	}
	Kokkos::fence();
}

struct bve_rh4{
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vorticity;

	bve_rh4(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity_) :
		xcos(xcos_), ycos(ycos_), zcos(zcos_), vorticity(vorticity_) {}

	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double theta, x, y, z, lat, lon, vor;
		for (int j = 0; j < jmax; j++) {
			x = xcos(i, j);
			y = ycos(i, j);
			z = zcos(i, j);
			xyz_to_latlon(lat, lon, x, y, z);
			// constant is 2Pi/7
			vor = 0.897597901025655211*Kokkos::sin(lat) + 30.0*Kokkos::sin(lat) * Kokkos::pow(Kokkos::cos(lat), 4) * Kokkos::cos(4*lon);
			vorticity(i,j) = vor / 86400.0; // convert vorticity from 1/day to 1/second
		}
	}
};

void bve_initialize(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity) {
	if (run_config.initial_condition == "rh4") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), bve_rh4(xcos, ycos, zcos, vorticity));
	}
}