#include "Kokkos_Core.hpp"
#include "run_config.hpp"
#include "general_utils.hpp"
#include "general_utils_impl.hpp"

struct bve_ssw_forcing {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vorticity;
	Kokkos::View<double**, Kokkos::LayoutRight> effective_vorticity;
	double time;
	double omega;

	bve_ssw_forcing(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_,  
					Kokkos::View<double**, Kokkos::LayoutRight>& vorticity_, Kokkos::View<double**, Kokkos::LayoutRight>& effective_vorticity_, double time_, double omega_) : xcos(xcos_), 
					ycos(ycos_), zcos(zcos_), vorticity(vorticity_), effective_vorticity(effective_vorticity_), time(time_), omega(omega_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double Tp = 4.0, Tf = 4.0 + 11.0, theta1 = Kokkos::numbers::pi / 3.0;
		double lat, lon, acomp, bcomp, ccomp;
		for (int j = 0; j < jmax; j++) {
			xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));
			if (time/86400.0 < Tp) {
				acomp = 0.5 * (1.0 - Kokkos::cos(Kokkos::numbers::pi * (time/86400.0) / Tp));
			} else if (time/86400.0 < Tf - Tp) {
				acomp = 1.0;
			} else if (time/86400.0 < Tf) {
				acomp = 0.5 * (1.0 - Kokkos::cos(Kokkos::numbers::pi + Kokkos::numbers::pi * ((time/86400.0) - Tf + Tp) / Tp));
			} else {
				acomp = 0.0;
			}
			if (lat > 0) {
				ccomp = Kokkos::pow(Kokkos::tan(theta1), 2) / Kokkos::pow(Kokkos::tan(lat), 2);
				bcomp = ccomp * Kokkos::exp(1.0 - ccomp);
			}
			effective_vorticity(i,j) = vorticity(i,j) + 0.6 * omega * acomp * bcomp * Kokkos::cos(lon);
		}
	}
};

void bve_forcing(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, 
						Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
						Kokkos::View<double**, Kokkos::LayoutRight>& zcos, 
						Kokkos::View<double**, Kokkos::LayoutRight>& vorticity, 
						Kokkos::View<double**, Kokkos::LayoutRight>& effective_vorticity, double time) {
	if (run_config.forcing) {
		if (run_config.forcing_type == "ssw") {
			Kokkos::parallel_for(run_config.active_panel_count, bve_ssw_forcing(xcos, ycos, zcos, vorticity, effective_vorticity, time, run_config.omega));
		}
	} else {
		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {vorticity.extent_int(0), vorticity.extent_int(1)}), copy_kokkos_view_2(effective_vorticity, vorticity));
	}
}