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

struct ones_2 {
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

struct sh43_2 {
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

struct coslon_2 {
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

struct sinlat_2 {
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

struct bve_rotation {
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vorticity;

	bve_rotation(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity_) :
			xcos(xcos_), ycos(ycos_), zcos(zcos_), vorticity(vorticity_) {}

	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double x, y, z, lat, lon;
		for (int j = 0; j < jmax; j++) {
			x = xcos(i,j);
			y = ycos(i,j);
			z = zcos(i,j);
			xyz_to_latlon(lat, lon, x, y, z);
			vorticity(i,j) = z / 86400.0;
		}
	}
};

struct bve_rh4 {
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

struct bve_gv {
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vorticity;

	bve_gv(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity_) :
			xcos(xcos_), ycos(ycos_), zcos(zcos_), vorticity(vorticity_) {}

	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double center_lat = 0.05 * M_PI;
		double center_lon = M_PI;
		double cx, cy, cz;
		xyz_from_lonlat(cx, cy, cz, center_lon, center_lat);
		double dx, dy, dz, d2;
		for (int j = 0; j < jmax; j++) {
			dx = xcos(i,j) - cx;
			dy = ycos(i,j) - cy;
			dz = zcos(i,j) - cz;
			d2 = dx*dx + dy*dy + dz*dz;
			vorticity(i,j) = 12.566370614359172 * exp(-16.0 * d2) / 86400.0; // constant is 4pi
		}
	}
};

struct bve_polar_vortex {
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vorticity;

	bve_polar_vortex(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, 
			Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity_) :
			xcos(xcos_), ycos(ycos_), zcos(zcos_), vorticity(vorticity_) {}

	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double theta0 = 15.0 * M_PI / 32.0, beta = 1.5;
		double lon, lat, vor;
		for (int j = 0; j < jmax; j++) {
			xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));
			vor = 2 * cos(lat) * pow(beta, 2) * sin(theta0 - lat) + sin(lat);
			vor *= M_PI * exp(-2 * pow(beta, 2) * (1 - cos(theta0 - lat)));
			vorticity(i,j) = vor / 86400.0;
		}
	}
};

struct tracer_vor {
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vorticity;
	Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace> passive_tracers;
	int index;

	tracer_vor(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity_,
				Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace>& passive_tracers_,
				int index_) : vorticity(vorticity_), passive_tracers(passive_tracers_), index(index_) {}

	void operator()(const int i) const {
		int jmax = passive_tracers.extent_int(1);
		for (int j = 0; j < jmax; j++) {
			passive_tracers(i,j,index) = vorticity(i,j);
		}
	}
};

struct tracer_z {
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace> passive_tracers;
	int index;

	tracer_z(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_,
				Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace>& passive_tracers_,
				int index_) : zcos(zcos_), passive_tracers(passive_tracers_), index(index_) {}

	void operator()(const int i) const {
		int jmax = passive_tracers.extent_int(1);
		for (int j = 0; j < jmax; j++) {
			passive_tracers(i,j,index) = zcos(i,j);
		}
	}
};

struct integrate_vorticity {
	using value_type = double;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vorticity;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> area;

	integrate_vorticity(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area_) :
						vorticity(vorticity_), area(area_) {}

	void operator()(const int i, value_type& update) const {
		int jmax = vorticity.extent_int(1);
		for (int j = 0; j < jmax; j++) {
			update += vorticity(i,j) * area(i,j);
		}
	}

	void join(value_type& dst, const value_type& src) const {
		dst += src;
	}

	void init(value_type& dst) const {
		dst = 0;
	}
};

struct modify_vor {
	double discrepancy;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vorticity;

	modify_vor(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity_, double discrepancy_) : vorticity(vorticity_), discrepancy(discrepancy_) {}

	void operator()(const int i) const {
		int jmax = vorticity.extent_int(1);
		for (int j = 0; j < jmax; j++) {
			vorticity(i,j) -= discrepancy / 12.566370614359172; // constant is 4pi
		}
	}
};

void bve_initialize(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area) {
	if (run_config.initial_condition == "rh4") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), bve_rh4(xcos, ycos, zcos, vorticity));
	} else if (run_config.initial_condition == "gv") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), bve_gv(xcos, ycos, zcos, vorticity));
	} else if (run_config.initial_condition == "pv") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), bve_polar_vortex(xcos, ycos, zcos, vorticity));
	} else if (run_config.initial_condition == "rot") {
		Kokkos::parallel_for(
			Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), bve_rotation(xcos, ycos, zcos, vorticity));
	}

	double integrated_vor = 0;
	if (run_config.balance_ic) {
		Kokkos::parallel_reduce(Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), integrate_vorticity(vorticity, area), integrated_vor);
		if (std::abs(integrated_vor) > 1e-15) {
			Kokkos::parallel_for(
				Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), modify_vor(vorticity, integrated_vor));
		}
	}
}

void tracer_initialize(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity, 
						Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace>& passive_tracers) {
	for (int i = 0; i < run_config.tracer_count; i++) {
		if (run_config.tracers[i] == "vor") {
			Kokkos::parallel_for(Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), 
							tracer_vor(vorticity, passive_tracers, i));
		} else if (run_config.tracers[i] == "z") {
			Kokkos::parallel_for(Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, run_config.active_panel_count), 
							tracer_z(zcos, passive_tracers, i));
		}
	}
}

struct swe_test_case_2 {
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vors;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> divs;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> height;

	swe_test_case_2(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vors_, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& divs_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& height_) : 
					xcos(xcos_), ycos(ycos_), zcos(zcos_), vors(vors_), divs(divs_), height(height_) {}

	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		// double alpha = 0.0; // rotation 
		double lat, lon;
		double u0 = M_PI/(6.0*86400.0); // 2pi/(12 days), converted to 1/s 
		// double g = 9.81 / 6371000.0; // 9.81 m/s^2, normalized by Earth radius
		for (int j = 0; j < jmax; j++) {
			xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));
			vors(i,j) = 2 * u0 * sin(lat);
			divs(i,j) = 0.0;
			height(i,j) = 2.94e4 / (6371000.0*6371000.0) - (2.0*M_PI*M_PI/(6.0*86400.0*86400.0) + 0.5 * u0 * u0) * sin(lat) * sin(lat);
			// height(i,j) = zcos(i,j);
		}
	}
};

struct swe_test_case_5 {
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vors;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> divs;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> height;

	swe_test_case_5(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vors_, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& divs_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& height_) : 
					xcos(xcos_), ycos(ycos_), zcos(zcos_), vors(vors_), divs(divs_), height(height_) {}

	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		// double alpha = 0.0; // rotation 
		double lat, lon;
		double u0 = 20.0 / 6371000.0; // 20 m/s, normalized by Earth radius
		// double g = 9.81 / 6371000.0; // 9.81 m/s^2, normalized by Earth radius
		for (int j = 0; j < jmax; j++) {
			xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));
			vors(i,j) = 2 * u0 * sin(lat);
			divs(i,j) = 0.0;
			height(i,j) = 5960.0*9.81 / (6371000.0*6371000.0) - (u0*M_PI*M_PI/(86400.0*86400.0) + 0.5 * u0 * u0) * sin(lat) * sin(lat);
		}
	}
};

void swe_initialize(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vorticity, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& divergence, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& height, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area) {
	if (run_config.initial_condition == "tc2") {
		Kokkos::parallel_for(run_config.active_panel_count, swe_test_case_2(xcos, ycos, zcos, vorticity, divergence, height));
	} else if (run_config.initial_condition == "tc5") {
		Kokkos::parallel_for(run_config.active_panel_count, swe_test_case_2(xcos, ycos, zcos, vorticity, divergence, height));
	}
}