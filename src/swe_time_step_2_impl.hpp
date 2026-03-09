#ifndef H_SWE_TIME_STEP_2_IMPL_H
#define H_SWE_TIME_STEP_2_IMPL_H

#include <Kokkos_Core.hpp>
#include "run_config.hpp"
#include "deriv_funcs_impl.hpp"
#include "interp_funcs_impl.hpp"
#include <iostream>

struct swe_tendency_computation_2{
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z;
	Kokkos::View<double**, Kokkos::LayoutRight> vor;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend;
	Kokkos::View<double**, Kokkos::LayoutRight> div;
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend;
	Kokkos::View<double**, Kokkos::LayoutRight> height;
	Kokkos::View<double**, Kokkos::LayoutRight> surface;
	Kokkos::View<double**, Kokkos::LayoutRight> height_tend;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	double omega;
	int degree;
	int offset;

	swe_tendency_computation_2(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vor_, Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend_, Kokkos::View<double**, Kokkos::LayoutRight>& div_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& div_tend_, Kokkos::View<double**, Kokkos::LayoutRight>& height_, Kokkos::View<double**, Kokkos::LayoutRight>& surface_, Kokkos::View<double**, Kokkos::LayoutRight>& height_tend_, 
				Kokkos::View<CubedSpherePanel*> cubed_sphere_panels_, double omega_, int degree_, int offset_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), 
				vor(vor_), vor_tend(vor_tend_), div(div_), div_tend(div_tend_), height(height_), surface(surface_), height_tend(height_tend_), cubed_sphere_panels(cubed_sphere_panels_), omega(omega_), degree(degree_), offset(offset_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double min_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i+offset).min_xi;
		double max_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i+offset).max_xi;
		double min_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i+offset).min_eta;
		double max_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i+offset).max_eta;
		// double u0 = (Kokkos::numbers::pi) / (6.0*86400.0);
		double u0 = 20.0 / 6371000.0;
		double xi_vel[121], eta_vel[121], curl_rad_comp[121], lap_comp[121], dp, height_lap[121], div_comp_xi[121], div_comp_eta[121], height_div[121], zetaf;
		double dp_vals[121], dp_lap[121], uvel, vvel, x, y, z, xc, yc, zc, vx, vy, vz;
		double lat, lon;
		for (int j = 0; j < xcos.extent_int(1); j++) {

			
			x = xcos(i,j);
			y = ycos(i,j);
			z = zcos(i,j);
			xyz_to_latlon(lat, lon, x, y, z);
			// if (abs(z) < 1 - 1e-16) { // away from pole
			// 	uvel = (-y*xc + x*yc)/Kokkos::sqrt(x*x+y*y);
			// 	vvel = -((x*xc+y*yc)*z-(x*x+y*y)*zc)/Kokkos::sqrt(x*x+y*y);
			// } else {
			// 	uvel = 0;
			// 	vvel = 0;
			// }
			// uvel = -u0*Kokkos::cos(lat);
			// if (Kokkos::abs(vvel) < 1e-9) {
			// 	vvel = 0;
			// }
			// vvel = 0;
			// vx = -Kokkos::sin(lat)*Kokkos::cos(lon)*vvel - Kokkos::sin(lon)*uvel;
			// vy = -Kokkos::sin(lat)*Kokkos::sin(lon)*vvel + Kokkos::cos(lon)*uvel;
			// vz = Kokkos::cos(lat)*vvel;
			vx = vel_x(i,j);
			vy = vel_y(i,j);
			vz = vel_z(i,j);
			xietavec_from_xyzvec(xi_vel[j], eta_vel[j], vx, vy, vz, xcos(i,j), ycos(i,j), zcos(i,j));
			div_comp_xi[j] = height(i,j)*xi_vel[j];
			div_comp_eta[j] = height(i,j)*eta_vel[j];
			zetaf = vor(i,j) + 2 * omega * zcos(i,j);
			xi_vel[j] *= zetaf;
			eta_vel[j] *= zetaf;
			// dp = vel_x(i,j)*vel_x(i,j) + vel_y(i,j)*vel_y(i,j) + vel_z(i,j)*vel_z(i,j);
			dp = vx*vx + vy*vy + vz*vz;
			lap_comp[j] = surface(i,j);
			// - (5960.0 / 6371000.0 - 1.0/(9.81/6371000.0)*(u0*Kokkos::numbers::pi*Kokkos::numbers::pi/(86400.0*86400.0) + 0.5 * u0 * u0) * Kokkos::sin(lat) * Kokkos::sin(lat));
			dp_vals[j] = 0.5*dp;
			
		}
		single_panel_curl_rad_comp_fd(curl_rad_comp, xi_vel, eta_vel, min_xi, max_xi, min_eta, max_eta, degree);
		single_panel_lap_fd(height_lap, lap_comp, min_xi, max_xi, min_eta, max_eta, degree);
		single_panel_lap_fd(dp_lap, dp_vals, min_xi, max_xi, min_eta, max_eta, degree);
		// single_panel_curl_rad_comp(curl_rad_comp, xi_vel, eta_vel, min_xi, max_xi, min_eta, max_eta, degree);
		// single_panel_lap_no_filter(height_lap, lap_comp, min_xi, max_xi, min_eta, max_eta, degree);
		// single_panel_lap(dp_lap, dp_vals, min_xi, max_xi, min_eta, max_eta, degree);
		// single_panel_div(height_div, div_comp_xi, div_comp_eta, min_xi, max_xi, min_eta, max_eta, degree);
		single_panel_div_fd(height_div, div_comp_xi, div_comp_eta, min_xi, max_xi, min_eta, max_eta, degree);

		double abs_vor, abs_vor_tend, grad_comp, vel_u;
		for (int j = 0; j < xcos.extent_int(1); j++) {
			x = xcos(i,j);
			y = ycos(i,j);
			z = zcos(i,j);
			xyz_to_latlon(lat, lon, x, y, z);
			abs_vor = vor(i,j) + 2.0 * omega * z;
			abs_vor_tend = -div(i,j) * abs_vor;
			vor_tend(i,j) = abs_vor_tend - 2.0 * omega * vel_z(i,j);
			height_lap[j] *= 9.81/6371000.0;
			// height_lap[j] *= 0.5;
			// curl_rad_comp[j] *= 0.5;
			// curl_rad_comp[j] = -2.0*(u0+omega)*u0*(Kokkos::cos(lat)*Kokkos::cos(lat)-2.0*Kokkos::sin(lat)*Kokkos::sin(lat));
			// dp_lap[j] = u0*u0*(2*Kokkos::sin(lat)*Kokkos::sin(lat)-Kokkos::cos(lat)*Kokkos::cos(lat));
			// height_lap[j] = (u0+2*omega)*u0*(2*Kokkos::sin(lat)*Kokkos::sin(lat)-Kokkos::cos(lat)*Kokkos::cos(lat));
			div_tend(i,j) = curl_rad_comp[j] - height_lap[j] - dp_lap[j]; 
			height_tend(i,j) = height_div[j];
			if (1.0 - Kokkos::abs(z) < 1e-5) {
				div_tend(i,j) = 0.0;
				height_tend(i,j) = 0.0;
			}
			// if ((i == 1) and (j == 1)) {
			// 	std::cout << "div tend: " << div_tend(i,j) << std::endl;
			// 	std::cout << "curl rad comp: " << curl_rad_comp[j] << std::endl;
			// 	std::cout << "curl rad comp: " << -2.0*(u0+omega)*u0*(Kokkos::cos(lat)*Kokkos::cos(lat)-2.0*Kokkos::sin(lat)*Kokkos::sin(lat)) << std::endl;
			// 	std::cout << "dp lap comp: " << dp_lap[j] << std::endl;
			// 	std::cout << "dp lap comp: " << u0*u0*(2*Kokkos::sin(lat)*Kokkos::sin(lat)-Kokkos::cos(lat)*Kokkos::cos(lat)) << std::endl;
			// 	std::cout << "surface lap comp: " << height_lap[j] << std::endl;
			// 	std::cout << "surface lap comp: " << (u0+2*omega)*u0*(2*Kokkos::sin(lat)*Kokkos::sin(lat)-Kokkos::cos(lat)*Kokkos::cos(lat)) << std::endl;
			// }
		}


	}
};

#endif
