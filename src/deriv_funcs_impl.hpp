#ifndef H_DERIV_FUNCS_IMPL_H
#define H_DERIV_FUNCS_IMPL_H

#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
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
		// fixed eta value
		for (int i = 0; i < degree+1; j++) {
			index = j * (degree+1) + i;
			deriv_vals[index] = 0;
			fi = func_vals[index];
			wi = bli_weights[i];
			xi = cheb_xi[i];
			for (int k = 0; k < degree+1; k++) {
				if (i != k) {
					index2 = j * (degree+1) + k;
					deriv_vals[index] += (func_vals[index2]-fi) * bli_weights[k] / (xi - cheb_xi[k]);
				}
			}
			deriv_vals[index] /= wi;
		}
	}
}

KOKKOS_INLINE_FUNCTION
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
			index = j * (degree+1) + i;
			deriv_vals[index] = 0;
			fi = func_vals[index];
			wi = bli_weights[i];
			yi = cheb_eta[i];
			for (int k = 0; k < degree+1; k++) {
				if (j != k) {
					index2 = j * (degree+1) + k;
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

	KOKKOS_INLINE_FUNCTION
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
			}
		}
	}
};

struct panel_laplacian {
	Kokkos::View<double**, Kokkos::LayoutRight> laplacian_vals;
	Kokkos::View<double**, Kokkos::LayoutRight> func_vals;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int degree;
	int offset;

	panel_laplacian(Kokkos::View<double**, Kokkos::LayoutRight>& laplacian_vals_, Kokkos::View<double**, Kokkos::LayoutRight>& func_vals_, 
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, int degree_, int offset_) : laplacian_vals(laplacian_vals_), 
					func_vals(func_vals_), cubed_sphere_panels(cubed_sphere_panels_), degree(degree_), offset(offset_) {}

	KOKKOS_INLINE_FUNCTION
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
		double xi_derivs[pc], eta_derivs[pc], xixi_derivs[pc], xieta_derivs[pc], etaeta_derivs[pc];
		bli_deriv_xi(xi_derivs, &func_vals(i-offset,0), degree, min_xi, max_xi);
		bli_deriv_eta(eta_derivs, &func_vals(i-offset, 0), degree, min_eta, max_eta);
		bli_deriv_xi(xixi_derivs, xi_derivs, degree, min_xi, max_xi);
		bli_deriv_xi(xieta_derivs, eta_derivs, degree, min_xi, max_xi);
		bli_deriv_eta(etaeta_derivs, eta_derivs, degree, min_eta, max_eta);
		double xi, eta;
		double cheb_points[degree+1];
		for (int j = 0; j < degree+1; j++) {
			cheb_points[j] = Kokkos::cos(Kokkos::numbers::pi * j / degree);
		}
		int index;
		double X, Y, C2, D2, delta;
		for (int j = 0; j < degree+1; j++) { // xi loop
			xi = cheb_points[j] * xi_scale + xi_offset;
			for (int k = 0; k < degree+1; k++) { // eta loop
				eta = cheb_points[k] * eta_scale + eta_offset;
				index = j * (degree+1)+k;
				X = Kokkos::tan(xi);
				Y = Kokkos::tan(eta);
				C2 = 1+X*X;
				D2 = 1+Y*Y;
				delta = 1+X*X+Y*Y;
				laplacian_vals(i-offset,index) = delta*(xixi_derivs[index]/C2+etaeta_derivs[index]/D2+2.0*X*Y/(C2*D2)*xieta_derivs[index]);
			}
		}
	}
};

#endif
