#ifndef H_INTERP_H
#define H_INTERP_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "initialize_cubed_sphere.hpp"
#include "cubed_sphere_transforms.hpp"

void interp_vals_bli(double* basis_vals, double xi, double eta, double min_xi, double max_xi, 
						double min_eta, double max_eta, int interp_deg);

void interp_init(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_info);

int point_locate_panel(Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, double x, double y, double z);

// int point_locate_panel2(Kokkos::View<CubedSpherePanel*> cubed_sphere_panels, double x, double y, double z);

template <class LayoutType> struct latlon_interp {
	Kokkos::View<double**, LayoutType, Kokkos::HostSpace> data;
	Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace> cubed_sphere_panels;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> interp_data;
	Kokkos::View<double*, Kokkos::HostSpace> lats;
	Kokkos::View<double*, Kokkos::HostSpace> lons;
	int degree;
	int offset;

	latlon_interp(Kokkos::View<double**, LayoutType, Kokkos::HostSpace>& data_, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels_, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_data_, Kokkos::View<double*, Kokkos::HostSpace>& lats_, 
					Kokkos::View<double*, Kokkos::HostSpace>& lons_, int degree_, int offset_) : data(data_), cubed_sphere_panels(cubed_sphere_panels_), interp_data(interp_data_), 
					lats(lats_), lons(lons_), degree(degree_), offset(offset_) {}

	void operator()(const int i, const int j) const {
		// interpolates data to lat[i] lon[j]
		double lat = lats(i) * M_PI / 180.0;
		double lon = lons(j) * M_PI / 180.0;
		double colat = M_PI / 2.0 - lat;
		double x, y, z;
		int one_d_index;
		x = sin(colat) * cos(lon);
		y = sin(colat) * sin(lon);
		z = cos(colat);
		int panel = point_locate_panel(cubed_sphere_panels, x, y, z);
		double xi, eta, xieta[2], bli_basis_vals[121];
		xieta_from_xyz(x, y, z, xieta);
		interp_vals_bli(bli_basis_vals, xieta[0], xieta[1], M_PI/4.0*cubed_sphere_panels(panel).min_xi, M_PI/4.0*cubed_sphere_panels(panel).max_xi, 
						M_PI/4.0*cubed_sphere_panels(panel).min_eta, M_PI/4.0*cubed_sphere_panels(panel).max_eta, degree);
		interp_data(i,j) = 0;
		for (int m = 0; m < degree+1; m++) {
			for (int n = 0; n < degree+1; n++) {
				one_d_index = m * (degree + 1) + n;
				interp_data(i,j) += bli_basis_vals[one_d_index] * data(panel-offset,one_d_index);
			}
		}
	}
};

template <class LayoutType> void interp_to_latlon(const RunConfig& run_config, Kokkos::View<double**, LayoutType, Kokkos::HostSpace>& data, 
								Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interped_data, 
								Kokkos::View<double*, Kokkos::HostSpace>& lats, Kokkos::View<double*, Kokkos::HostSpace>& lons) {
	int offset = cubed_sphere_panels.extent_int(0) - run_config.active_panel_count;
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultHostExecutionSpace(), {0, 0}, {lats.extent_int(0), lons.extent_int(0)}), latlon_interp<LayoutType>(data, cubed_sphere_panels, interped_data, lats, lons, run_config.interp_degree, offset));
}

#endif
