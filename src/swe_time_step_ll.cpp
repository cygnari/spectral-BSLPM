#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "initialize_octo_sphere.hpp"
#include "fmm_funcs.hpp"
#include "fmm_interactions/swe_vel.hpp"
#include "forcing_funcs.hpp"
#include "general_utils_impl.hpp"

struct vor_tend_ll{
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z;
	Kokkos::View<double**, Kokkos::LayoutRight> vor;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend;
	Kokkos::View<double**, Kokkos::LayoutRight> div;
	double omega;

	vor_tend_ll(Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_, Kokkos::View<double**, Kokkos::LayoutRight>& vor_, Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& div_, double omega_) : zcos(zcos_), vel_z(vel_z_), vor(vor_), vor_tend(vor_tend_), div(div_), omega(omega_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		double abs_vor = vor(i,j) + 2.0 * omega * zcos(i,j);
		double abs_vor_tend = -abs_vor * div(i,j);
		vor_tend(i,j) = abs_vor_tend - 2.0 * omega * vel_z(i,j);
	}
};

struct height_tend_ll{
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> u_vel;
	Kokkos::View<double**, Kokkos::LayoutRight> v_vel;
	Kokkos::View<double**, Kokkos::LayoutRight> height;
	Kokkos::View<double**, Kokkos::LayoutRight> height_tend;
	double grid_dx;

	height_tend_ll(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& u_vel_, Kokkos::View<double**, Kokkos::LayoutRight>& v_vel_, Kokkos::View<double**, Kokkos::LayoutRight>& height_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& height_tend_, double grid_dx_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), u_vel(u_vel_), v_vel(v_vel_), height(height_), height_tend(height_tend_), grid_dx(grid_dx_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		double lat, lon, div1, div2;
		int panel_index, one_d_index, min_lat, max_lat, lon_count, jp, jm;
		xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));
		height_tend(i,j) = 0;

		double grid_dx_rad = grid_dx*Kokkos::numbers::pi/180.0;

		if (j+1 == xcos.extent_int(1)) {
			jp = 0;
		} else {
			jp = j+1;
		} 
		if (j-1 == -1) {
			jm = xcos.extent_int(1)-1;
		} else {
			jm = j-1;
		}

		div1 = (height(i,jp)*u_vel(i,jp) - height(i,jm)*u_vel(i,jm))/(2.0*grid_dx); // lon div component
		min_lat = 0;
		max_lat = xcos.extent_int(0);
		lon_count = xcos.extent_int(1);
		if (i == min_lat) {
			// stencil based on -0.5 0 1, but -0.5 is at the south pole and cos(-pi/2) = 0
			div2 = (3.0*height(i,j)*v_vel(i,j)*Kokkos::cos(lat)+height(i+1,j)*v_vel(i+1,j)*Kokkos::cos(lat+grid_dx_rad))/(3.0*grid_dx_rad);
		} else if (i == max_lat-1) {
			// stencil based on -1 0 0.5, but 0.5 is at the north pole and cos(pi/2) = 0
			div2 = (-height(i-1,j)*v_vel(i-1,j)*Kokkos::cos(lat-grid_dx_rad)-3.0*height(i,j)*v_vel(i,j)*Kokkos::cos(lat))/(3.0*grid_dx_rad);
		} else {
			div2 = (height(i+1,j)*v_vel(i+1,j)*Kokkos::cos(lat+grid_dx_rad) - height(i-1,j)*v_vel(i-1,j)*Kokkos::cos(lat-grid_dx_rad))/(2.0*grid_dx_rad);
		}
		
		// if (1.0-Kokkos::abs(zcos(i,j)) < 1e-8) {
		// 	// close to pole
		// 	height_tend(i,j) = 0;
		// } else {
		height_tend(i,j) = -(div1+div2) / Kokkos::cos(lat);
		// }
		
	}
};

struct div_tend_ll{
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> u_vel;
	Kokkos::View<double**, Kokkos::LayoutRight> v_vel;
	Kokkos::View<double**, Kokkos::LayoutRight> vor;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_dp;
	Kokkos::View<double**, Kokkos::LayoutRight> height;
	Kokkos::View<double**, Kokkos::LayoutRight> topo;
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend;
	double grid_dx;
	double omega;

	div_tend_ll(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& u_vel_, Kokkos::View<double**, Kokkos::LayoutRight>& v_vel_, Kokkos::View<double**, Kokkos::LayoutRight>& vor_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_dp_, 
				Kokkos::View<double**, Kokkos::LayoutRight>& height_, Kokkos::View<double**, Kokkos::LayoutRight>& topo_, Kokkos::View<double**, Kokkos::LayoutRight>& div_tend_, 
				double grid_dx_, double omega_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), u_vel(u_vel_), v_vel(v_vel_), vor(vor_), vel_dp(vel_dp_), height(height_), topo(topo_), div_tend(div_tend_), grid_dx(grid_dx_), omega(omega_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		double lat, lon, val1, val2, val3, val4, curl_rad_comp, lap_comp, lap_val, f_val, lc1, lc2, val5;
		int panel_index, one_d_index, min_lat, max_lat, jp, jm, jopp;
		// std::cout << i << " " << j << std::endl;
		// std::cout << xcos(i,j) << " " << ycos(i,j) << zcos(i,j) << std::endl;
		xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));
		// std::cout << lat << " " << lon << std::endl;
		double grid_dx_rad = grid_dx*Kokkos::numbers::pi/180.0;

		if (j+1 == xcos.extent_int(1)) {
			jp = 0;
		} else {
			jp = j+1;
		} 
		if (j-1 == -1) {
			jm = xcos.extent_int(1)-1;
		} else {
			jm = j-1;
		}
		// double dp = xcos(i,jp) * xcos(i,j) + ycos(i,jp) * ycos(i,j) + zcos(i,jp) * zcos(i,j);
		// double grid_dx_lon = Kokkos::acos(dp);
		// std::cout << jm << " " << jp << std::endl;
		val1 = (vor(i,jp)+2.0*omega*zcos(i,jp))*v_vel(i,jp);
		val2 = (vor(i,jm)+2.0*omega*zcos(i,jm))*v_vel(i,jm);
		val3 = 9.81/6371000.0*(height(i,jp)+topo(i,jp)) + 0.5*vel_dp(i,jp);
		val4 = 9.81/6371000.0*(height(i,jm)+topo(i,jm)) + 0.5*vel_dp(i,jm);
		f_val = 9.81/6371000.0*(height(i,j)+topo(i,j)) + 0.5*vel_dp(i,j);

		curl_rad_comp = (val1-val2) / (2.0*grid_dx_rad);
		// std::cout << curl_rad_comp << std::endl;

		min_lat = 0;
		max_lat = xcos.extent_int(0);
		
		lap_comp = (val3 - 2.0*f_val + val4) / (grid_dx_rad*grid_dx_rad*Kokkos::cos(lat)*Kokkos::cos(lat));
		// std::cout << lap_comp << std::endl;

		jopp = (j + xcos.extent_int(1) / 2) % xcos.extent_int(1);

		if (i == 0) {
			// southern most lat
			// for the curl rad comp, south pole value is multiplied by 0, but stencil is still based on -0.5 0 1
			val1 = (vor(i,j)+2.0*omega*zcos(i,j)*u_vel(i,j))*Kokkos::cos(lat);
			val2 = (vor(i+1,j)+2.0*omega*zcos(i+1,j)*u_vel(i+1,j))*Kokkos::cos(lat+grid_dx_rad);
			curl_rad_comp -= (3.0*val1+val2)/(3.0*grid_dx_rad);
			// lap_comp += curl_rad_comp / Kokkos::cos(lat);
			// val1 = (vor(i,jopp)-2.0*omega*zcos(i,jopp)*u_vel(i,jopp))*Kokkos::cos(lat);
			// val2 = (vor(i+1,j)-2.0*omega*zcos(i+1,j)*u_vel(i+1,j))*Kokkos::cos(lat+grid_dx_rad);
			// curl_rad_comp -= (val2 - val1)/(2.0*grid_dx_rad);
			lap_comp += curl_rad_comp / Kokkos::cos(lat);
			// val3 = 9.81/6371000.0*(height(i,jopp)+topo(i,jopp)) + 0.5*vel_dp(i,jopp);
			// val4 = 9.81/6371000.0*(height(i+1,j)+topo(i+1,j)) + 0.5*vel_dp(i+1,j);
			// val3 = 9.81/6371000.0*(height(i+1,j)+topo(i+1,j)) + 0.5*vel_dp(i+1,j);
			// val4 = 9.81/6371000.0*(height(i+2,j)+topo(i+2,j)) + 0.5*vel_dp(i+2,j);
			// val5 = 9.81/6371000.0*(height(i+3,j)+topo(i+3,j)) + 0.5*vel_dp(i+3,j);
			// lap_comp += -Kokkos::tan(lat)*(val4 - val3)/(2.0*grid_dx_rad) + (val3 - 2.0*f_val+val4) / (grid_dx_rad*grid_dx_rad);
			// lap_comp += -Kokkos::tan(lat)*(-11.0*f_val+18.0*val3-9.0*val4+2.0*val5) / (6.0*grid_dx_rad) + (2.0*f_val-5.0*val3+4.0*val4-val_5)/(grid_d)
			// div_tend(i,j) = 0;
		} else if (i == max_lat-1) {
			// northern most lat
			val1 = (vor(i,j)+2.0*omega*zcos(i,j)*u_vel(i,j))*Kokkos::cos(lat);
			val2 = (vor(i-1,j)+2.0*omega*zcos(i-1,j)*u_vel(i-1,j))*Kokkos::cos(lat-grid_dx_rad);
			curl_rad_comp -= (-3.0*val1-val2)/(3.0*grid_dx_rad);
			lap_comp += curl_rad_comp / Kokkos::cos(lat);
			// val1 = (vor(i,jopp)-2.0*omega*zcos(i,jopp)*u_vel(i,jopp))*Kokkos::cos(lat);
			// val2 = (vor(i-1,j)-2.0*omega*zcos(i-1,j)*u_vel(i-1,j))*Kokkos::cos(lat-grid_dx_rad);
			// curl_rad_comp -= (val1-val2) / (2.0*grid_dx_rad);
			// div_tend(i,j) = 0;
			// val3 = 9.81/6371000.0*(height(i,jopp)+topo(i,jopp)) + 0.5*vel_dp(i,jopp);
			// val4 = 9.81/6371000.0*(height(i-1,j)+topo(i-1,j)) + 0.5*vel_dp(i-1,j);
			// val3 = 9.81/6371000.0*(height(i-1,j)+topo(i-1,j)) + 0.5*vel_dp(i-1,j);
			// val4 = 9.81/6371000.0*(height(i-2,j)+topo(i-2,j)) + 0.5*vel_dp(i-2,j);
			// val5 = 9.81/6371000.0*(height(i-3,j)+topo(i-3,j)) + 0.5*vel_dp(i-3,j);
			// lap_comp += -Kokkos::tan(lat)*(val3-val4)/(2.0*grid_dx_rad) + (val3-2.0*f_val+val4)/(grid_dx_rad*grid_dx_rad);
		} else {
			val1 = (vor(i+1,j)+2.0*omega*zcos(i+1,j))*u_vel(i+1,j)*Kokkos::cos(lat+grid_dx_rad);
			val2 = (vor(i-1,j)+2.0*omega*zcos(i-1,j))*u_vel(i-1,j)*Kokkos::cos(lat-grid_dx_rad);
			curl_rad_comp -= (val1-val2) / (2.0*grid_dx_rad);
			val3 = 9.81/6371000.0*(height(i+1,j)+topo(i+1,j)) + 0.5*vel_dp(i+1,j);
			val4 = 9.81/6371000.0*(height(i-1,j)+topo(i-1,j)) + 0.5*vel_dp(i-1,j);
			// lc1 = (val3 - f_val) * Kokkos::cos(lat + 0.5*grid_dx_rad);
			// lc2 = (f_val - val4) * Kokkos::cos(lat - 0.5*grid_dx_rad);
			// lc1 = -Kokkos::tan(lat)*(val3-val4)/(2.0*grid_dx_rad);
			// lc2 = (val3-2.0*f_val+val4)/(grid_dx_rad*grid_dx_rad);
			// lap_comp += Kokkos::cos(lat)*(lc1 - lc2)/grid_dx_rad;
			lap_comp += -Kokkos::tan(lat)*(val3-val4)/(2.0*grid_dx_rad) + (val3-2.0*f_val+val4)/(grid_dx_rad*grid_dx_rad);
		}
		// std::cout << curl_rad_comp / Kokkos::cos(lat) << " " << lap_comp << std::endl;

		// if (j == 0) {
		// 	std::cout << lat << div_tend(i,j) << std::endl;
		// }

		div_tend(i,j) = (curl_rad_comp / Kokkos::cos(lat) - lap_comp);
		// div_tend(i,j) -= (-4.5e-10+6.05e-10*Kokkos::cos(lat)*Kokkos::cos(lat))/864.0;
		// if (i == min_lat) {
		// 	div_tend(i,j) = 0;
		// } else if (i == min_lat + 1) {
		// 	div_tend(i,j) = 0;
		// } else if (i == max_lat - 2) {
		// 	div_tend(i,j) = 0;
		// } else if (i == max_lat - 1) {
		// 	div_tend(i,j) = 0;
		// }
		if (Kokkos::abs(lat) * 180.0/M_PI > 60) {
			div_tend(i,j) *= Kokkos::cos(lat)*Kokkos::cos(lat);
		}
		// if (j == 0) {
		// 	std::cout << lat << div_tend(i,j) << std::endl;
		// }
		// if (i < 10) {
		// 	div_tend(i,j) = 0;
		// } else if (
		// 	i > max_lat - 11) {
		// 	div_tend(i,j) = 0;
		// }
	}
};

struct laplacian_diffusion {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> field;
	Kokkos::View<double**, Kokkos::LayoutRight> new_field;
	double dt;
	double grid_dx;
	// double epsilon;

	laplacian_diffusion(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& field_, Kokkos::View<double**, Kokkos::LayoutRight>& new_field_, double dt_, double grid_dx_) : 
						xcos(xcos_), ycos(ycos_), zcos(zcos_), field(field_), new_field(new_field_), dt(dt_), grid_dx(grid_dx_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		int lat_count = xcos.extent_int(0);
		double f, f1, f2, f3, f4, lat, lon, lap1, lap2, lap3;
		int jp, jm;
		double grid_dx_rad = grid_dx*Kokkos::numbers::pi/180.0;
		double coeff = 0.89/ 6371000.0; 
		// 1.9 works for 2 degree
		// 0.89 works for 1 degree
		new_field(i,j) = field(i,j);
		if ((i > 0) and (i < lat_count-1)) {
			if (j+1 == xcos.extent_int(1)) {
				jp = 0;
			} else {
				jp = j+1;
			} 
			if (j-1 == -1) {
				jm = xcos.extent_int(1)-1;
			} else {
				jm = j-1;
			}
			f = field(i,j);
			f1 = field(i,jp);
			f2 = field(i,jm);
			f3 = field(i+1,j);
			f4 = field(i-1,j);
			xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));
			lap1 = 1.0 / (grid_dx_rad*grid_dx_rad*Kokkos::cos(lat)*Kokkos::cos(lat))*(f1-2.0*f+f2);
			lap2 = Kokkos::tan(lat)*(f3 - f4)/(2.0*grid_dx_rad);
			lap3 = (f3-2.0*f+f4)/(grid_dx_rad*grid_dx_rad);
			new_field(i,j) += (lap1 - lap2 + lap3) * dt * coeff; 
			// multiply by 1.9 for 2 degree resolution
			// not stable at 1 degree resolution, increase?
		}
	}
};

struct field_averaging {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> field;
	Kokkos::View<double**, Kokkos::LayoutRight> new_field;
	double epsilon;

	field_averaging(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& field_, Kokkos::View<double**, Kokkos::LayoutRight>& new_field_, double epsilon_) : 
						xcos(xcos_), ycos(ycos_), zcos(zcos_), field(field_), new_field(new_field_), epsilon(epsilon_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		int lat_count = xcos.extent_int(0);
		double f, f1, f2, f3, f4, lat, lon, lap1, lap2, lap3, coeff;
		int jp, jm, jopp;
		// double epsilon = 0.16;
		// double eps2 = 0.16;
		double eps2 = 0.00;
		
		xyz_to_latlon(lat, lon, xcos(i,0), ycos(i,0), zcos(i,0));
		if (Kokkos::abs(lat) > Kokkos::numbers::pi / 3.0) { // for TC5
		// if (Kokkos::abs(lat) > 70.0 * Kokkos::numbers::pi / 180.0) { // for galewsky
			f1 = epsilon*field(i,jp);
			f2 = epsilon*field(i,jm);
			if (j+1 == xcos.extent_int(1)) {
				jp = 0;
			} else {
				jp = j+1;
			} 
			if (j-1 == -1) {
				jm = xcos.extent_int(1)-1;
			} else {
				jm = j-1;
			}
			if ((i > 0) and (i < lat_count-1)) {
				f3 = epsilon*field(i+1,j);
				f4 = epsilon*field(i-1,j);
				coeff = 4;
			} else if (i == 0) {
				jopp = (j + xcos.extent_int(1) / 2) % xcos.extent_int(1);
				f3 = epsilon*field(i+1,j);
				// f4 = epsilon*field(i,jopp);
				f4 = 0;
				coeff = 3;
			} else {
				jopp = (j + xcos.extent_int(1) / 2) % xcos.extent_int(1);
				f3 = epsilon*field(i-1,j);
				// f4 = epsilon*field(i,jopp);
				coeff = 3;
				f4 = 0;
			}
			f = (1.0-coeff*epsilon)*field(i,j);
			new_field(i,j) = f + f1 + f2 + f3 + f4;
			new_field(i,j) *= Kokkos::cos(lat) * Kokkos::cos(lat);
		} else {
			// new_field(i,j) = field(i,j);
			f1 = eps2*field(i,jp);
			f2 = eps2*field(i,jm);
			if (j+1 == xcos.extent_int(1)) {
				jp = 0;
			} else {
				jp = j+1;
			} 
			if (j-1 == -1) {
				jm = xcos.extent_int(1)-1;
			} else {
				jm = j-1;
			}
			if ((i > 0) and (i < lat_count-1)) {
				f3 = eps2*field(i+1,j);
				f4 = eps2*field(i-1,j);
				coeff = 4;
			} else if (i == 0) {
				jopp = (j + xcos.extent_int(1) / 2) % xcos.extent_int(1);
				f3 = eps2*field(i+1,j);
				f4 = 0;
				coeff = 3;
			} else {
				jopp = (j + xcos.extent_int(1) / 2) % xcos.extent_int(1);
				f3 = eps2*field(i-1,j);
				coeff = 3;
				f4 = 0;
			}
			f = (1.0-coeff*eps2)*field(i,j);
			new_field(i,j) = f + f1 + f2 + f3 + f4;
			// new_field(i,j) *= Kokkos::cos(lat) * Kokkos::cos(lat);
		}
	}
};

void swe_tendency_computation_ll(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& vors, Kokkos::View<double**, Kokkos::LayoutRight>& divs, Kokkos::View<double**, Kokkos::LayoutRight>& height, 
							Kokkos::View<double**, Kokkos::LayoutRight>& topo, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, Kokkos::View<double**, Kokkos::LayoutRight> vor_update, Kokkos::View<double**, Kokkos::LayoutRight> div_update, Kokkos::View<double**, Kokkos::LayoutRight> height_update) {
	// computes tendencies with derivatives computed using finite differences
	Kokkos::View<double**, Kokkos::LayoutRight> vel_u ("u vel", run_config.lat_count, run_config.lon_count);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_v ("v vel", run_config.lat_count, run_config.lon_count);
	Kokkos::View<double**, Kokkos::LayoutRight> vel_dp ("vel dp", run_config.lat_count, run_config.lon_count);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_u.extent_int(0), vel_u.extent_int(1)}), zero_out(vel_u));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_v.extent_int(0), vel_v.extent_int(1)}), zero_out(vel_v));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_dp.extent_int(0), vel_dp.extent_int(1)}), zero_out(vel_dp));
	Kokkos::parallel_for(run_config.lat_count, xyz_vel_to_uv_vel(xcos, ycos, zcos, vel_x, vel_y, vel_z, vel_u, vel_v));
	// Kokkos::parallel_for(run_config.lat_count, filter_vals(vel_v, 1e-13));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), vel_dot_prod(vel_dp, vel_x, vel_y, vel_z));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), vor_tend_ll(zcos, vel_z, vors, vor_update, divs, run_config.omega));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), height_tend_ll(xcos, ycos, zcos, vel_u, vel_v, height, height_update, run_config.grid_spacing));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}),
						div_tend_ll(xcos, ycos, zcos, vel_u, vel_v, vors, vel_dp, height, topo, div_update, run_config.grid_spacing, run_config.omega));
						// div_tend_lagrangian(xcos, ycos, zcos, vel_u, vel_v, vors, vel_dp, height, topo, div_update, run_config.grid_spacing, run_config.omega));
	Kokkos::parallel_for(run_config.lat_count, filter_vals(div_update, 2e-12));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), clamp_vals(div_update, 1e-9));
	// Kokkos::parallel_for(run_config.lat_count, filter_vals(height_update, 2e-11));
	// Kokkos::parallel_for(run_config.lat_count, filter_vals(vor_update, 1e-10));
}

struct swe_compute_arrival_values_ll {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos; 
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vor;
	Kokkos::View<double**, Kokkos::LayoutRight> div;
	Kokkos::View<double**, Kokkos::LayoutRight> height;
	Kokkos::View<double**, Kokkos::LayoutRight> a_vor;
	Kokkos::View<double**, Kokkos::LayoutRight> a_div;
	Kokkos::View<double**, Kokkos::LayoutRight> a_height;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z_1;
	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend_1;
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend_1;
	Kokkos::View<double**, Kokkos::LayoutRight> height_tend_1;
	double dt;
	double dx;

	swe_compute_arrival_values_ll(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_,
							Kokkos::View<double**, Kokkos::LayoutRight>& vor_, Kokkos::View<double**, Kokkos::LayoutRight>& div_, Kokkos::View<double**, Kokkos::LayoutRight>& height_,
							Kokkos::View<double**, Kokkos::LayoutRight>& a_vor_, Kokkos::View<double**, Kokkos::LayoutRight>& a_div_, Kokkos::View<double**, Kokkos::LayoutRight>& a_height_,
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_1_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_1_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_1_,
							Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend_1_, Kokkos::View<double**, Kokkos::LayoutRight>& div_tend_1_, Kokkos::View<double**, Kokkos::LayoutRight>& h_tend_1_, 
							double dt_, double dx_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), vor(vor_), div(div_), height(height_), vel_x_1(vel_x_1_), vel_y_1(vel_y_1_), vel_z_1(vel_z_1_), 
							vor_tend_1(vor_tend_1_), div_tend_1(div_tend_1_), height_tend_1(h_tend_1_), dt(dt_), a_vor(a_vor_), a_div(a_div_), a_height(a_height_), dx(dx_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		double vor1, x, y, z, lat, lon, latv, lonv, lat_d, lon_d, lat_coeffs[3], lon_coeffs[3], vor_vals[3][3], pole_val, k1x, k1y, k1z, h_vals[3][3], div_vals[3][3];
		int lat_index, lon_index, lon_count, lat_count, lon_m, lon_p;
		bool lat_vals_computed = false;
		double vor_scale = 1.0;
		double div_scale = 1.0;
		double h_scale = 1.0;
		double u_vel, v_vel, xc, yc, zc, vvals_one[3], hvals_one[3];
		double dx_rad = dx * M_PI / 180.0;

		x = xcos(i,j);
		y = ycos(i,j);
		z = zcos(i,j);
		xc = vel_x_1(i,j);
		yc = vel_y_1(i,j);
		zc = vel_z_1(i,j);
		if (abs(z) < 1 - 1e-16) { // away from pole
			u_vel = (-y*xc + x*yc)/Kokkos::sqrt(x*x+y*y);
			v_vel = -((x*xc+y*yc)*z-(x*x+y*y)*zc)/Kokkos::sqrt(x*x+y*y);
		} else {
			u_vel = 0;
			v_vel = 0;
		}

		if (Kokkos::abs(v_vel) < 1e-16) {
			// meridional velocity is small, interpolate in one d
			xyz_to_latlon(lat, lon, x, y, z);
			lon -= u_vel * dt;
			lon_d = 180.0/Kokkos::numbers::pi * lon;
			lon_d = std::fmod(lon_d + 360.0, 360.0);
			lon_index = Kokkos::floor(lon_d/dx);
			lon_count = xcos.extent_int(1);
			lon_m = ((lon_index - 1) + lon_count) % lon_count;
			lon_p = (lon_index + 1) % lon_count;
			lon_coeffs[0] = (lon_d - dx*(lon_index+0.5))*(lon_d - dx*(lon_index+1.5)) / (2.0*dx*dx);
			lon_coeffs[1] = -(lon_d - dx*(lon_index-0.5))*(lon_d - dx*(lon_index+1.5)) / (dx*dx);
			lon_coeffs[2] = (lon_d - dx*(lon_index-0.5))*(lon_d - dx*(lon_index+0.5)) / (2.0*dx*dx);

			vvals_one[0] = vor(i, lon_m)+vor_scale*dt*vor_tend_1(i, lon_m);
			vvals_one[1] = vor(i, lon_index)+vor_scale*dt*vor_tend_1(i, lon_index);
			vvals_one[2] = vor(i, lon_p)+vor_scale*dt*vor_tend_1(i, lon_p);
			hvals_one[0] = height(i,lon_m) - dt*h_scale*height(i,lon_m)*div(i,lon_m);
			hvals_one[1] = height(i,lon_index) - dt*h_scale*height(i,lon_index)*div(i,lon_index);
			hvals_one[2] = height(i,lon_p) - dt*h_scale*height(i,lon_p)*div(i,lon_p);

			a_vor(i,j) = 0;
			a_height(i,j) = 0;
			for (int l = 0; l < 3; l++) { // lon index
				a_vor(i,j) += vvals_one[l] * lon_coeffs[l];
				a_height(i,j) += hvals_one[l] * lon_coeffs[l];
			}
		} else {
			// two d interpolation
			// compute departure point 
			k1x = vel_x_1(i,j);
			k1y = vel_y_1(i,j);
			k1z = vel_z_1(i,j);

			x = xcos(i,j) - dt*k1x;
			y = ycos(i,j) - dt*k1y;
			z = zcos(i,j) - dt*k1z;
			
			project_to_sphere(x, y, z); // departure points
			xyz_to_latlon(lat, lon, x, y, z);
			lat_d = 180.0/Kokkos::numbers::pi * lat;
			lon_d = 180.0/Kokkos::numbers::pi * lon;
			lon_d = std::fmod(lon_d + 360.0, 360.0);
			lat_count = xcos.extent_int(0);
			lon_count = xcos.extent_int(1);
			lat_index = Kokkos::floor((lat_d+90.0)/dx);
			lon_index = Kokkos::floor(lon_d/dx);

			if (lat_index == 0) {
				if (lat_d > -90.0+0.5*dx) {
					lat_index += 1;
				} else {
					lat_coeffs[0] = (lat_d-(-90.0+dx*(lat_index+0.5)))*(lat_d-(-90.0+dx*(lat_index+1.5))) / (dx*dx*0.5*1.5);
					lat_coeffs[1] = -(lat_d-(-90.0))*(lat_d-(-90.0+dx*(lat_index+1.5))) / (dx*dx*0.5);
					lat_coeffs[2] = (lat_d-(-90.0))*(lat_d-(-90.0+dx*(lat_index+0.5))) / (dx*dx*1.5);
					lat_vals_computed = true;
				}
			}
			if (lat_index == lat_count - 1) {
				if (lat_d < 90.0-0.5*dx) {
					lat_index -= 1;
				} else {
					lat_coeffs[0] = (lat_d-(-90.0+dx*(lat_index+0.5)))*(lat_d-90.0) / (dx*dx*1.5);
					lat_coeffs[1] = -(lat_d-(-90.0+dx*(lat_index-0.5)))*(lat_d-90.0) / (dx*dx*0.5);
					lat_coeffs[2] = (lat_d-(-90.0+dx*(lat_index-0.5)))*(lat_d-(-90.0+dx*(lat_index+0.5))) / (dx*dx*0.5*1.5);
					lat_vals_computed = true;
				}
			} 
			if (not lat_vals_computed) {
				lat_coeffs[0] = (lat_d-(-90.0+dx*(lat_index+0.5)))*(lat_d-(-90.0+dx*(lat_index+1.5))) / (2.0*dx*dx);
				lat_coeffs[1] = -(lat_d-(-90.0+dx*(lat_index-0.5)))*(lat_d-(-90.0+dx*(lat_index+1.5))) / (dx*dx);
				lat_coeffs[2] = (lat_d-(-90.0+dx*(lat_index-0.5)))*(lat_d-(-90.0+dx*(lat_index+0.5))) / (2.0*dx*dx);
			}

			lon_coeffs[0] = (lon_d - dx*(lon_index+0.5))*(lon_d - dx*(lon_index+1.5)) / (2.0*dx*dx);
			lon_coeffs[1] = -(lon_d - dx*(lon_index-0.5))*(lon_d - dx*(lon_index+1.5)) / (dx*dx);
			lon_coeffs[2] = (lon_d - dx*(lon_index-0.5))*(lon_d - dx*(lon_index+0.5)) / (2.0*dx*dx);

			lon_m = ((lon_index - 1) + lon_count) % lon_count;
			lon_p = (lon_index + 1) % lon_count;

			if (lat_index == 0) {
				// south pole, average surrounding points
				pole_val = 0.0;
				for (int k = 0; k < lon_count; k++) {
					pole_val += vor(0,k) + vor_scale*dt*vor_tend_1(0,k);
				}
				pole_val /= lon_count;
				vor_vals[0][0] = pole_val;
				vor_vals[0][1] = pole_val;
				vor_vals[0][2] = pole_val;
				pole_val = 0.0;
				for (int k = 0; k < lon_count; k++) {
					pole_val += height(0,k) - h_scale*height(0,k)*div(0,k)*dt;
				}
				pole_val /= lon_count;
				h_vals[0][0] = pole_val;
				h_vals[0][1] = pole_val;
				h_vals[0][2] = pole_val;
			} else {
				vor_vals[0][0] = vor(lat_index-1, lon_m)+vor_scale*dt*vor_tend_1(lat_index-1, lon_m);
				vor_vals[0][1] = vor(lat_index-1, lon_index)+vor_scale*dt*vor_tend_1(lat_index-1, lon_index);
				vor_vals[0][2] = vor(lat_index-1, lon_p)+vor_scale*dt*vor_tend_1(lat_index-1, lon_p);
				h_vals[0][0] = height(lat_index-1,lon_m) - dt*h_scale*height(lat_index-1,lon_m)*div(lat_index-1,lon_m);
				h_vals[0][1] = height(lat_index-1,lon_index) - dt*h_scale*height(lat_index-1,lon_index)*div(lat_index-1,lon_index);
				h_vals[0][2] = height(lat_index-1,lon_p) - dt*h_scale*height(lat_index-1,lon_p)*div(lat_index-1,lon_p);
			}

			vor_vals[1][0] = vor(lat_index, lon_m)+vor_scale*dt*vor_tend_1(lat_index, lon_m);
			vor_vals[1][1] = vor(lat_index, lon_index)+vor_scale*dt*vor_tend_1(lat_index, lon_index);
			vor_vals[1][2] = vor(lat_index, lon_p)+vor_scale*dt*vor_tend_1(lat_index, lon_p);
			h_vals[1][0] = height(lat_index,lon_m) - dt*h_scale*height(lat_index,lon_m)*div(lat_index,lon_m);
			h_vals[1][1] = height(lat_index,lon_index) - dt*h_scale*height(lat_index,lon_index)*div(lat_index,lon_index);
			h_vals[1][2] = height(lat_index,lon_p) - dt*h_scale*height(lat_index,lon_p)*div(lat_index,lon_p);

			if (lat_index == lat_count-1) {
				// north pole, average surrounding points
				pole_val = 0.0;
				for (int k = 0; k < lon_count; k++) {
					pole_val += vor(lat_index,k) + vor_scale*dt*vor_tend_1(lat_index,k);
				}
				pole_val /= lon_count;
				vor_vals[2][0] = pole_val;
				vor_vals[2][1] = pole_val;
				vor_vals[2][2] = pole_val;
				pole_val = 0.0;
				for (int k = 0; k < lon_count; k++) {
					pole_val += height(lat_index,k) - h_scale*height(lat_index,k)*div(lat_index,k)*dt;
				}
				pole_val /= lon_count;
				h_vals[2][0] = pole_val;
				h_vals[2][1] = pole_val;
				h_vals[2][2] = pole_val;
			} else {
				vor_vals[2][0] = vor(lat_index+1, lon_m)+vor_scale*dt*vor_tend_1(lat_index+1, lon_m);
				vor_vals[2][1] = vor(lat_index+1, lon_index)+vor_scale*dt*vor_tend_1(lat_index+1, lon_index);
				vor_vals[2][2] = vor(lat_index+1, lon_p)+vor_scale*dt*vor_tend_1(lat_index+1, lon_p);
				h_vals[2][0] = height(lat_index+1,lon_m) - dt*h_scale*height(lat_index+1,lon_m)*div(lat_index+1,lon_m);
				h_vals[2][1] = height(lat_index+1,lon_index) - dt*h_scale*height(lat_index+1,lon_index)*div(lat_index+1,lon_index);
				h_vals[2][2] = height(lat_index+1,lon_p) - dt*h_scale*height(lat_index+1,lon_p)*div(lat_index+1,lon_p);
			}

			a_vor(i,j) = 0;
			a_height(i,j) = 0;
			for (int k = 0; k < 3; k++) { // lat index
				for (int l = 0; l < 3; l++) { // lon index
					a_vor(i,j) += vor_vals[k][l] * lat_coeffs[k] * lon_coeffs[l];
					a_height(i,j) += h_vals[k][l] * lat_coeffs[k] * lon_coeffs[l];
				}
			}
		}
		
		a_div(i,j) = div(i,j) + div_scale*dt*div_tend_1(i,j);
	}
};

struct polar_filtering {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos; 
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> field;
	double grid_dx;

	polar_filtering(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
						Kokkos::View<double**, Kokkos::LayoutRight>& field_, double grid_dx_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), field(field_), grid_dx(grid_dx_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		// fourier filters the values at latitude index i, only filters above 60 degrees
		double lat, lon;
		double sin_coeffs[30], cos_coeffs[30], val;
		int spec_comps;
		double grid_dx_rad = grid_dx * Kokkos::numbers::pi / 180.0;
		xyz_to_latlon(lat, lon, xcos(i,0), ycos(i,0), zcos(i,0));
		// if (Kokkos::abs(lat) > 70.0 * Kokkos::numbers::pi / 180.0) {  // maybe for galewsky?
		if (Kokkos::abs(lat) > Kokkos::numbers::pi / 3.0) { // works for TC5
			spec_comps = Kokkos::floor(90.0 - Kokkos::abs(lat*180.0/Kokkos::numbers::pi));
			spec_comps = Kokkos::min(spec_comps + 1, 29);
			for (int j = 0; j < 30; j++) {
				sin_coeffs[0] = 0;
				cos_coeffs[0] = 0;
			}
			for (int j = 0; j < xcos.extent_int(1); j++) {
				xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));
				// if (field(i,j) != field(i,j)) {
				// 	field(i,j) = 0.0;
				// }
				if (field(i,j) > 1e-11) {
					field(i,j) = 1e-11;
				} else if (field(i,j) < -1e-11) {
					field(i,j) = 1e-11;
				}
				cos_coeffs[0] += field(i,j);
				for (int k = 1; k < spec_comps; k++) {
					cos_coeffs[k] += field(i,j) * Kokkos::cos(k*lat) * grid_dx_rad;
					sin_coeffs[k] += field(i,j) * Kokkos::sin(k*lat) * grid_dx_rad;
				}
			}
			cos_coeffs[0] /= xcos.extent_int(1);
			for (int k = 1; k < spec_comps; k++) {
				cos_coeffs[k] /= xcos.extent_int(1);
				sin_coeffs[k] /= xcos.extent_int(1);
			}
			for (int k = 0; k < spec_comps; k++) {
				if (cos_coeffs[k] != cos_coeffs[k]) {
					cos_coeffs[k] = 0;
				} 
				if (sin_coeffs[k] != sin_coeffs[k]) {
					sin_coeffs[k] = 0;
				} 
			}
			for (int j = 0; j < xcos.extent_int(1); j++) {
				val = 0;
				val = cos_coeffs[0];
				xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));
				for (int k = 1; k < spec_comps; k++) {
					val += Kokkos::cos(k*lat)*cos_coeffs[k] + Kokkos::sin(k*lat)*sin_coeffs[k];
				}
				// if (field(i,j) > val) {
				// 	field(i,j) = val;
				// }
				field(i,j) = val;
			}

		}
	}
};

struct combine_states {
	Kokkos::View<double**, Kokkos::LayoutRight> x1;
	Kokkos::View<double**, Kokkos::LayoutRight> x2;
	Kokkos::View<double**, Kokkos::LayoutRight> x3;
	double coeff;

	combine_states(Kokkos::View<double**, Kokkos::LayoutRight>& x1_, Kokkos::View<double**, Kokkos::LayoutRight>& x2_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& x3_, double coeff_) : x1(x1_), x2(x2_), x3(x3_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		x1(i,j) = coeff * x2(i,j) + (1.0-coeff) * x3(i,j);
	}
};

void swe_vel_tend_computation(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& area, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vors, Kokkos::View<double**, Kokkos::LayoutRight>& divs, Kokkos::View<double**, Kokkos::LayoutRight>& height, 
							Kokkos::View<double**, Kokkos::LayoutRight>& topo, Kokkos::View<CubedSpherePanel*>& sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, Kokkos::View<double**, Kokkos::LayoutRight>& vor_tend, Kokkos::View<double**, Kokkos::LayoutRight>& div_tend, Kokkos::View<double**, Kokkos::LayoutRight>& height_tend) {
	// compute the velocity and vorticity/divergence/height tendencies
	int dim2size = (run_config.interp_degree+1)*(run_config.interp_degree+1);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_vors ("proxy source vors", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_divs ("proxy source divs", run_config.panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_vors.extent_int(0), proxy_source_vors.extent_int(1)}), zero_out(proxy_source_vors));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_source_divs.extent_int(0), proxy_source_divs.extent_int(1)}), zero_out(proxy_source_divs));
	upward_pass_ll(run_config, xcos, ycos, zcos, sphere_panels, area, vors, proxy_source_vors, leaf_panel_points);
	upward_pass_ll(run_config, xcos, ycos, zcos, sphere_panels, area, divs, proxy_source_divs, leaf_panel_points);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_1("proxy target vels", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_2("proxy target vels", run_config.panel_count, dim2size);
	Kokkos::View<double**, Kokkos::LayoutRight> proxy_target_3("proxy target vels", run_config.panel_count, dim2size);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_1.extent_int(0), proxy_target_1.extent_int(1)}), zero_out(proxy_target_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_2.extent_int(0), proxy_target_2.extent_int(1)}), zero_out(proxy_target_2));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {proxy_target_3.extent_int(0), proxy_target_3.extent_int(1)}), zero_out(proxy_target_3));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_x.extent_int(0), vel_x.extent_int(1)}), zero_out(vel_x));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_y.extent_int(0), vel_y.extent_int(1)}), zero_out(vel_y));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vel_z.extent_int(0), vel_z.extent_int(1)}), zero_out(vel_z));
	swe_vel_interactions_ll(run_config, xcos, ycos, zcos, proxy_target_1, proxy_target_2, proxy_target_3, vel_x, vel_y, vel_z, proxy_source_vors, proxy_source_divs, vors, divs, area, interaction_list, sphere_panels, leaf_panel_points);
	downward_pass_3_ll(run_config, xcos, ycos, zcos, sphere_panels, proxy_target_1, proxy_target_2, proxy_target_3, vel_x, vel_y, vel_z, leaf_panel_points);
	MPI_Allreduce(MPI_IN_PLACE, &vel_x(0,0), run_config.lat_count*run_config.lon_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_y(0,0), run_config.lat_count*run_config.lon_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vel_z(0,0), run_config.lat_count*run_config.lon_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	Kokkos::View<double**, Kokkos::LayoutRight> vor_update_1 ("vorticity k1", run_config.lat_count, run_config.lon_count);
	Kokkos::View<double**, Kokkos::LayoutRight> div_update_1 ("divergence k1", run_config.lat_count, run_config.lon_count);
	Kokkos::View<double**, Kokkos::LayoutRight> height_update_1 ("height k1", run_config.lat_count, run_config.lon_count);
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {vor_update_1.extent_int(0), vor_update_1.extent_int(1)}), zero_out(vor_update_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {div_update_1.extent_int(0), div_update_1.extent_int(1)}), zero_out(div_update_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {height_update_1.extent_int(0), height_update_1.extent_int(1)}), zero_out(height_update_1));

	swe_tendency_computation_ll(run_config, xcos, ycos, zcos, vors, divs, height, topo, vel_x, vel_y, vel_z, vor_update_1, div_update_1, height_update_1);
	Kokkos::parallel_for(run_config.lat_count, polar_filtering(xcos, ycos, zcos, height_update_1, run_config.grid_spacing));

	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {vor_tend.extent_int(0), vor_tend.extent_int(1)}), copy_kokkos_view_2(vor_tend, vor_update_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {div_tend.extent_int(0), div_tend.extent_int(1)}), copy_kokkos_view_2(div_tend, div_update_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {height_tend.extent_int(0), height_tend.extent_int(1)}), copy_kokkos_view_2(height_tend, height_update_1));
}

void swe_back_rk4_step_ll(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
							Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& area, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vors, Kokkos::View<double**, Kokkos::LayoutRight>& divs, Kokkos::View<double**, Kokkos::LayoutRight>& height, 
							// Kokkos::View<double**, Kokkos::LayoutRight>& topo, Kokkos::View<OctoSpherePanel*>& sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<double**, Kokkos::LayoutRight>& topo, Kokkos::View<CubedSpherePanel*>& sphere_panels, Kokkos::View<interact_pair*>& interaction_list, 
							Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, 
							Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, double time) {
	// SWE single SSPRK3 step lat lon coordinates
	// u1

	Kokkos::View<double**, Kokkos::LayoutRight> new_vors ("new vor", run_config.lat_count, run_config.lon_count);
	Kokkos::View<double**, Kokkos::LayoutRight> new_divs_1 ("new divs", run_config.lat_count, run_config.lon_count);
	Kokkos::View<double**, Kokkos::LayoutRight> new_divs_2_1 ("new divs 2", run_config.lat_count, run_config.lon_count);
	Kokkos::View<double**, Kokkos::LayoutRight> new_divs_3_1 ("new divs 3", run_config.lat_count, run_config.lon_count);
	Kokkos::View<double**, Kokkos::LayoutRight> new_h ("new heights", run_config.lat_count, run_config.lon_count);

	Kokkos::View<double**, Kokkos::LayoutRight> vor_tend_1 ("vor tend 1", vors.extent_int(0), vors.extent_int(1));
	Kokkos::View<double**, Kokkos::LayoutRight> div_tend_1 ("div tend 1", vors.extent_int(0), vors.extent_int(1));
	Kokkos::View<double**, Kokkos::LayoutRight> hei_tend_1 ("height tend 1", vors.extent_int(0), vors.extent_int(1));

	swe_vel_tend_computation(run_config, xcos, ycos, zcos, area, vors, divs, height, topo, sphere_panels, interaction_list, leaf_panel_points, vel_x, vel_y, vel_z, vor_tend_1, div_tend_1, hei_tend_1);

	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
							swe_compute_arrival_values_ll(xcos, ycos, zcos, vors, divs, height, new_vors, new_divs_1, new_h, vel_x, vel_y, vel_z, vor_tend_1, div_tend_1, hei_tend_1, run_config.delta_t, run_config.grid_spacing));

	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
							field_averaging(xcos, ycos, zcos, new_divs_1, new_divs_2_1, 0.16));
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
							laplacian_diffusion(xcos, ycos, zcos, new_divs_2_1, new_divs_3_1, run_config.delta_t, run_config.grid_spacing));

	Kokkos::parallel_for(run_config.lat_count, polar_filtering(xcos, ycos, zcos, new_divs_3_1, run_config.grid_spacing));

	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {vors.extent_int(0), vors.extent_int(1)}), copy_kokkos_view_2(vors, new_vors));
	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {divs.extent_int(0), divs.extent_int(1)}), copy_kokkos_view_2(divs, new_divs_3_1));
	Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {height.extent_int(0), height.extent_int(1)}), copy_kokkos_view_2(height, new_h));

	// // u2

	// Kokkos::View<double**, Kokkos::LayoutRight> new_vors_2 ("new vor", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> new_divs_2 ("new divs", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> new_divs_2_2 ("new divs 2", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> new_divs_3_2 ("new divs 3", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> new_h_2 ("new heights", run_config.lat_count, run_config.lon_count);

	// Kokkos::View<double**, Kokkos::LayoutRight> vor_tend_2 ("vor tend 2", vors.extent_int(0), vors.extent_int(1));
	// Kokkos::View<double**, Kokkos::LayoutRight> div_tend_2 ("div tend 2", vors.extent_int(0), vors.extent_int(1));
	// Kokkos::View<double**, Kokkos::LayoutRight> hei_tend_2 ("height tend 2", vors.extent_int(0), vors.extent_int(1));

	// Kokkos::View<double**, Kokkos::LayoutRight> vel_x_2 ("vel x 2", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> vel_y_2 ("vel y 2", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> vel_z_2 ("vel z 2", run_config.lat_count, run_config.lon_count);

	// swe_vel_tend_computation(run_config, xcos, ycos, zcos, area, new_vors, new_divs_3_1, new_h, topo, sphere_panels, interaction_list, leaf_panel_points, vel_x_2, vel_y_2, vel_z_2, vor_tend_2, div_tend_2, hei_tend_2);
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						swe_compute_arrival_values_ll(xcos, ycos, zcos, new_vors, new_divs_3_1, new_h, new_vors_2, new_divs_2, new_h_2, vel_x_2, vel_y_2, vel_z_2, vor_tend_2, div_tend_2, hei_tend_2, run_config.delta_t, run_config.grid_spacing));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						field_averaging(xcos, ycos, zcos, new_divs_2, new_divs_2_2, 0.16));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						laplacian_diffusion(xcos, ycos, zcos, new_divs_2_2, new_divs_3_2, run_config.delta_t, run_config.grid_spacing));

	// Kokkos::parallel_for(run_config.lat_count, polar_filtering(xcos, ycos, zcos, new_divs_3_2, run_config.grid_spacing));

	// Kokkos::View<double**, Kokkos::LayoutRight> vor_state_2 ("new vor 2", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> div_state_2 ("new divs 2", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> hei_state_2 ("new heights 2", run_config.lat_count, run_config.lon_count);

	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						combine_states(vor_state_2, vors, new_vors_2, 0.75));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						combine_states(div_state_2, divs, new_divs_3_2, 0.75));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						combine_states(hei_state_2, height, new_h_2, 0.75));

	// // u n+1

	// Kokkos::View<double**, Kokkos::LayoutRight> new_vors_3 ("new vor", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> new_divs_3 ("new divs", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> new_divs_2_3 ("new divs 2", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> new_divs_3_3 ("new divs 3", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> new_h_3 ("new heights", run_config.lat_count, run_config.lon_count);

	// Kokkos::View<double**, Kokkos::LayoutRight> vor_tend_3 ("vor tend 3", vors.extent_int(0), vors.extent_int(1));
	// Kokkos::View<double**, Kokkos::LayoutRight> div_tend_3 ("div tend 3", vors.extent_int(0), vors.extent_int(1));
	// Kokkos::View<double**, Kokkos::LayoutRight> hei_tend_3 ("height tend 3", vors.extent_int(0), vors.extent_int(1));

	// Kokkos::View<double**, Kokkos::LayoutRight> vel_x_3 ("vel x 3", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> vel_y_3 ("vel y 3", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> vel_z_3 ("vel z 3", run_config.lat_count, run_config.lon_count);

	// Kokkos::View<double**, Kokkos::LayoutRight> vor_state_3 ("new vor 3", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> div_state_3 ("new divs 3", run_config.lat_count, run_config.lon_count);
	// Kokkos::View<double**, Kokkos::LayoutRight> hei_state_3 ("new heights 3", run_config.lat_count, run_config.lon_count);

	// swe_vel_tend_computation(run_config, xcos, ycos, zcos, area, vor_state_2, div_state_2, hei_state_2, topo, sphere_panels, interaction_list, leaf_panel_points, vel_x_3, vel_y_3, vel_z_3, vor_tend_3, div_tend_3, hei_tend_3);
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						swe_compute_arrival_values_ll(xcos, ycos, zcos, vor_state_2, div_state_2, hei_state_2, new_vors_3, new_divs_3, new_h_3, vel_x_3, vel_y_3, vel_z_3, vor_tend_3, div_tend_3, hei_tend_3, run_config.delta_t, run_config.grid_spacing));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						field_averaging(xcos, ycos, zcos, new_divs_3, new_divs_2_3, 0.16));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						laplacian_diffusion(xcos, ycos, zcos, new_divs_2_3, new_divs_3_3, run_config.delta_t, run_config.grid_spacing));
	// Kokkos::parallel_for(run_config.lat_count, polar_filtering(xcos, ycos, zcos, new_divs_3_3, run_config.grid_spacing));

	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						combine_states(vor_state_3, vors, new_vors_3, 1.0/3.0));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						combine_states(div_state_3, vors, new_divs_3_3, 1.0/3.0));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {run_config.lat_count, run_config.lon_count}), 
	// 						combine_states(hei_state_3, vors, new_h_3, 1.0/3.0));

	// Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {vors.extent_int(0), vors.extent_int(1)}), copy_kokkos_view_2(vors, vor_state_3));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {divs.extent_int(0), divs.extent_int(1)}), copy_kokkos_view_2(divs, div_state_3));
	// Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {height.extent_int(0), height.extent_int(1)}), copy_kokkos_view_2(height, hei_state_3));
}









