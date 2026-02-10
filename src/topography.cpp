#include <Kokkos_Core.hpp>
#include "run_config.hpp"
#include "general_utils_impl.hpp"
#include <iostream>

struct cone_mountain {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> disp_x;
	Kokkos::View<double**, Kokkos::LayoutRight> disp_y;
	Kokkos::View<double**, Kokkos::LayoutRight> disp_z;
	Kokkos::View<double**, Kokkos::LayoutRight> height;
	Kokkos::View<double**, Kokkos::LayoutRight> eff_height;

	cone_mountain(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& disp_x_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_z_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& height_, Kokkos::View<double**, Kokkos::LayoutRight>& eff_height_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), 
					disp_x(disp_x_), disp_y(disp_y_), disp_z(disp_z_), height(height_), eff_height(eff_height_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = xcos.extent_int(1);
		double radius = Kokkos::numbers::pi / 9.0;
		// double radius = Kokkos::numbers::pi;
		double height0 = 2000.0 / 6371000.0; // 2000 meters, normalized by earth radius
		double lambdac = 1.5*Kokkos::numbers::pi;
		double thetac = Kokkos::numbers::pi / 6.0;
		double lat, lon, r, x, y, z;
		for (int j = 0; j < jmax; j++) {
			x = xcos(i,j) + disp_x(i,j);
			y = ycos(i,j) + disp_y(i,j);
			z = zcos(i,j) + disp_z(i,j);
			project_to_sphere(x, y, z);
			xyz_to_latlon(lat, lon, x, y, z);
			lon += 2*Kokkos::numbers::pi;
			lon = Kokkos::fmod(lon, 2*Kokkos::numbers::pi);
			r = Kokkos::sqrt(Kokkos::fmin((lon-lambdac)*(lon-lambdac) + (lat-thetac)*(lat-thetac), radius*radius));
			eff_height(i,j) = height(i,j) + height0 * (1.0 - r / radius);
		}
	}
};

void apply_topography(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, Kokkos::View<double**, Kokkos::LayoutRight>& zcos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& disp_x, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y, Kokkos::View<double**, Kokkos::LayoutRight>& disp_z, 
							Kokkos::View<double**, Kokkos::LayoutRight>& height, Kokkos::View<double**, Kokkos::LayoutRight>& effective_height) {
	if (run_config.topo_type == "cone") {
		Kokkos::parallel_for(run_config.active_panel_count, cone_mountain(xcos, ycos, zcos, disp_x, disp_y, disp_z, height, effective_height));
	} else {
		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {height.extent_int(0), height.extent_int(1)}), copy_kokkos_view_2(effective_height, height));
	}
}

struct cone_mountain_2 {
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> disp_x;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> disp_y;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> disp_z;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> height;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> eff_height;

	cone_mountain_2(Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& disp_x_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& disp_y_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& disp_z_, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& height_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& eff_height_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), 
					disp_x(disp_x_), disp_y(disp_y_), disp_z(disp_z_), height(height_), eff_height(eff_height_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		// std::cout << 
		int jmax = xcos.extent_int(1);
		double radius = Kokkos::numbers::pi / 9.0;
		// double radius = Kokkos::numbers::pi;
		// double radius = Kokkos::numbers::pi;
		double height0 = 2000.0 / 6371000.0; // 2000 meters, normalized by earth radius
		// double height0 = 1.0;
		double lambdac = 1.5*Kokkos::numbers::pi;
		double thetac = Kokkos::numbers::pi / 6.0;
		double lat, lon, r, x, y, z;
		for (int j = 0; j < jmax; j++) {
			x = xcos(i,j) + disp_x(i,j);
			y = ycos(i,j) + disp_y(i,j);
			z = zcos(i,j) + disp_z(i,j);
			project_to_sphere(x, y, z);
			xyz_to_latlon(lat, lon, x, y, z);
			lon += 2*Kokkos::numbers::pi;
			lon = Kokkos::fmod(lon, 2*Kokkos::numbers::pi);
			r = std::sqrt(std::fmin((lon-lambdac)*(lon-lambdac) + (lat-thetac)*(lat-thetac), radius*radius));
			eff_height(i,j) = height(i,j) + height0 * (1.0 - r / radius);
		}
	}
};

void apply_topography_2(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& disp_x, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& disp_y, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& disp_z, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& height, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& effective_height) {
	if (run_config.topo_type == "cone") {
		Kokkos::parallel_for(run_config.active_panel_count, cone_mountain_2(xcos, ycos, zcos, disp_x, disp_y, disp_z, height, effective_height));
	} 
	// else {
	// 	// effective_height = height;
	// }
}