#include <cmath>
#include <Kokkos_Core.hpp>
#include <iostream>

#include "cubed_sphere_transforms_impl.hpp"
#include "general_utils_impl.hpp"
#include "initialize_cubed_sphere.hpp"
#include "run_config.hpp"

void bli_deriv_xi(double* deriv_vals, double* func_vals, int degree, double min_xi, double max_xi) {
	// computes the xi derivative using barycentric Lagrange differentiation
	double cheb_xi[degree+1], bli_weights[degree+1], xi_range, xi_offset;
	xi_range = 0.5*(max_xi - min_xi);
	xi_offset = 0.5*(max_xi + min_xi);

	for (int i = 0; i < degree+1; i++) {
		cheb_xi[i] = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_range + xi_offset;
		if (i == 0) {
			bli_weights[i] = 0.5;
		} else if (i == degree) {
			bli_weights[i] = 0.5 * Kokkos::pow(-1, i);
		} else {
			bli_weights[i] = Kokkos::pow(-1, i);
		}
	}

	int index, index2;
	double fi, wi, xi;
	for (int j = 0; j < degree+1; j++) {
	// for (int j = 0; j < 1; j++) {
		// fixed eta value
		for (int i = 0; i < degree+1; i++) {
			index = i * (degree+1) + j;
			deriv_vals[index] = 0;
			fi = func_vals[index];
			wi = bli_weights[i];
			xi = cheb_xi[i];
			// std::cout << fi << " " << wi << " " << xi << std::endl;
			for (int k = 0; k < degree+1; k++) {
				if (i != k) {
					index2 = k * (degree+1) + j;
					// std::cout << (func_vals[index2]-fi) * bli_weights[k] / (xi - cheb_xi[k]) << std::endl;
					// std::cout << func_vals[index2] << " " << fi << " " << bli_weights[k] << " " << xi << " " << cheb_xi[k] << std::endl;
					deriv_vals[index] += (func_vals[index2]-fi) * bli_weights[k] / (xi - cheb_xi[k]);
				}
			}
			deriv_vals[index] /= wi;
			// std::cout << deriv_vals[index] << std::endl;
		}
	}
}

void bli_deriv_eta(double* deriv_vals, double* func_vals, int degree, double min_eta, double max_eta) {
	// computes the xi derivative using barycentric Lagrange differentiation
	double cheb_eta[degree+1], bli_weights[degree+1], eta_range, eta_offset;
	eta_range = 0.5*(max_eta - min_eta);
	eta_offset = 0.5*(max_eta + min_eta);

	for (int i = 0; i < degree+1; i++) {
		cheb_eta[i] = Kokkos::cos(Kokkos::numbers::pi * i / degree) * eta_range + eta_offset;
		if (i == 0) {
			bli_weights[i] = 0.5;
		} else if (i == degree) {
			bli_weights[i] = 0.5 * Kokkos::pow(-1, i);
		} else {
			bli_weights[i] = Kokkos::pow(-1, i);
		}
	}

	int index, index2;
	double fi, wi, yi;
	for (int i = 0; i < degree+1; i++) {
		// fixed xi value
		for (int j = 0; j < degree+1; j++) {
			index = i * (degree+1) + j;
			deriv_vals[index] = 0;
			fi = func_vals[index];
			wi = bli_weights[j];
			yi = cheb_eta[j];
			// std::cout << fi << " " << wi << " " << yi << std::endl;
			for (int k = 0; k < degree+1; k++) {
				if (j != k) {
					index2 = i * (degree+1) + k;
					// std::cout << func_vals[index2] << " " << fi << " " << bli_weights[k] << " " << yi << " " << cheb_eta[k] << std::endl;
					// std::cout << (func_vals[index2]-fi) * bli_weights[k] / (yi - cheb_eta[k]) << std::endl;
					deriv_vals[index] += (func_vals[index2]-fi) * bli_weights[k] / (yi - cheb_eta[k]);
				}
			}
			deriv_vals[index] /= wi;
		}
	}
}

struct panel_gradient {
	Kokkos::View<double**, Kokkos::LayoutRight> x_comps;
	Kokkos::View<double**, Kokkos::LayoutRight> y_comps;
	Kokkos::View<double**, Kokkos::LayoutRight> z_comps;
	Kokkos::View<double**, Kokkos::LayoutRight> func_vals;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int degree;
	int offset;

	panel_gradient(Kokkos::View<double**, Kokkos::LayoutRight>& x_comps_, Kokkos::View<double**, Kokkos::LayoutRight>& y_comps_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& z_comps_, Kokkos::View<double**, Kokkos::LayoutRight>& func_vals_, 
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, int degree_, int offset_) : x_comps(x_comps_), y_comps(y_comps_), 
					z_comps(z_comps_), func_vals(func_vals_), cubed_sphere_panels(cubed_sphere_panels_), degree(degree_), offset(offset_) {}

	void operator()(const int i) const {
		double min_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).min_eta;
		double max_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).max_eta;
		double min_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).min_xi;
		double max_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).max_xi;
		double xi_offset = 0.5*(min_xi + max_xi);
		double xi_scale = 0.5*(max_xi - min_xi);
		double eta_offset = 0.5*(min_eta + max_eta);
		double eta_scale = 0.5*(max_eta - min_eta);
		int pc = (degree+1) * (degree+1);
		double xi_derivs[pc], eta_derivs[pc];
		bli_deriv_xi(xi_derivs, &func_vals(i-offset,0), degree, min_xi, max_xi);
		bli_deriv_eta(eta_derivs, &func_vals(i-offset, 0), degree, min_eta, max_eta);
		double cheb_points[degree+1];
		for (int j = 0; j < degree+1; j++) {
			cheb_points[j] = Kokkos::cos(Kokkos::numbers::pi * j / degree);
		}
		int index;
		double xi, eta, lon_deriv, colat_deriv, x, y, z, xyz[3], colat, lon;
		double X, Y;
		for (int j = 0; j < degree+1; j++) { // xi loop
			xi = cheb_points[j] * xi_scale + xi_offset;
			for (int k = 0; k < degree+1; k++) { // eta loop
				index = j * (degree+1) + k;
				eta = cheb_points[k] * eta_scale + eta_offset;
				xyzvec_from_xietavec(x_comps(i-offset, index), y_comps(i-offset, index), z_comps(i-offset, index), xi_derivs[index], eta_derivs[index], cubed_sphere_panels(i).face, xi, eta);
				// loncolatvec_from_xietavec(lon_deriv, colat_deriv, xi_derivs[index], eta_derivs[index], cubed_sphere_panels(i).face, xi, eta);
				// xyz_from_xieta(xi, eta, cubed_sphere_panels(i).face, xyz);
				// x = xyz[0];
				// y = xyz[1];
				// z = xyz[2];
				// xyzvec_from_loncolatvec(x_comps(i-offset, index), y_comps(i-offset, index), z_comps(i-offset, index), lon_deriv, colat_deriv, x,y,z);
			}
		}
	}
};

void xyz_gradient(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& x_comps, 
					Kokkos::View<double**, Kokkos::LayoutRight>& y_comps, Kokkos::View<double**, Kokkos::LayoutRight>& z_comps, 
					Kokkos::View<double**, Kokkos::LayoutRight>& func_vals, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels) {
	int ub = run_config.panel_count;
	int lb = run_config.panel_count - run_config.active_panel_count;
	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), panel_gradient(x_comps, y_comps, z_comps, func_vals, cubed_sphere_panels, run_config.interp_degree, lb));
}