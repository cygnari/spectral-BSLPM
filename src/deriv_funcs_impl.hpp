#ifndef H_DERIV_FUNCS_IMPL_H
#define H_DERIV_FUNCS_IMPL_H

#include <Kokkos_Core.hpp>
#include <iostream>
#include "cubed_sphere_transforms_impl.hpp"

KOKKOS_INLINE_FUNCTION
void bli_deriv_xi(double* deriv_vals, double* func_vals, int degree, double min_xi, double max_xi) {
	// computes the xi derivative using barycentric Lagrange differentiation
	double xi_range, xi_offset;
	xi_range = 0.5*(max_xi - min_xi);
	xi_offset = 0.5*(max_xi + min_xi);

	int index, index2;
	double fi, wi, xi, xi2, wi2;
	for (int j = 0; j < degree+1; j++) {
		// fixed eta value
		for (int i = 0; i < degree+1; i++) { // outer xi loop
			// index = j * (degree+1) + i;
			index = i * (degree+1) + j;
			deriv_vals[index] = 0;
			fi = func_vals[index];
			if (i == 0) {
				wi = 0.5;
			} else if (i == degree) {
				wi = 0.5 * Kokkos::pow(-1, i);
			} else {
				wi = Kokkos::pow(-1, i);
			}
			xi = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_range + xi_offset;
			for (int k = 0; k < degree+1; k++) { // inner xi loop
				if (i != k) {
					index2 = k * (degree+1) + j;
					xi2 = Kokkos::cos(Kokkos::numbers::pi * k / degree) * xi_range + xi_offset;
					if (k == 0) {
						wi2 = 0.5;
					} else if (k == degree) {
						wi2 = 0.5 * Kokkos::pow(-1, k);
					} else {
						wi2 = Kokkos::pow(-1, k);
					}
					deriv_vals[index] += (func_vals[index2]-fi) * wi2 / (xi - xi2);
				}
			}
			deriv_vals[index] /= wi;
		}
	}
}

KOKKOS_INLINE_FUNCTION
void bli_deriv_eta(double* deriv_vals, double* func_vals, int degree, double min_eta, double max_eta) {
	// computes the xi derivative using barycentric Lagrange differentiation
	double eta_range, eta_offset;
	eta_range = 0.5*(max_eta - min_eta);
	eta_offset = 0.5*(max_eta + min_eta);

	int index, index2;
	double fi, wi, yi, wi2, yi2;
	for (int i = 0; i < degree+1; i++) {
		// fixed xi value
		for (int j = 0; j < degree+1; j++) { // outer eta loop
			index = i * (degree+1) + j;
			// index = i * (degree+1)
			deriv_vals[index] = 0;
			fi = func_vals[index];
			if (j == 0) {
				wi = 0.5;
			} else if (j == degree) {
				wi = 0.5 * Kokkos::pow(-1, j);
			} else {
				wi = Kokkos::pow(-1, j);
			}
			yi = Kokkos::cos(Kokkos::numbers::pi * j / degree) * eta_range + eta_offset;
			for (int k = 0; k < degree+1; k++) { // inner eta loop
				if (j != k) {
					index2 = i * (degree+1) + k;
					if (k == 0) {
						wi2 = 0.5;
					} else if (k == degree) {
						wi2 = 0.5 * Kokkos::pow(-1, k);
					} else {
						wi2 = Kokkos::pow(-1, k);
					}
					yi2 = Kokkos::cos(Kokkos::numbers::pi * k / degree) * eta_range + eta_offset;
					deriv_vals[index] += (func_vals[index2]-fi) * wi2 / (yi - yi2);
				}
			}
			deriv_vals[index] /= wi;
		}
	}
}

struct panel_gradient {
	Kokkos::View<double**, Kokkos::LayoutStride> x_comps;
	Kokkos::View<double**, Kokkos::LayoutStride> y_comps;
	Kokkos::View<double**, Kokkos::LayoutStride> z_comps;
	Kokkos::View<double**, Kokkos::LayoutRight> func_vals;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int degree;
	int offset;

	panel_gradient(Kokkos::View<double**, Kokkos::LayoutStride>& x_comps_, Kokkos::View<double**, Kokkos::LayoutStride>& y_comps_, 
					Kokkos::View<double**, Kokkos::LayoutStride>& z_comps_, Kokkos::View<double**, Kokkos::LayoutRight>& func_vals_, 
					Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, int degree_, int offset_) : x_comps(x_comps_), y_comps(y_comps_), 
					z_comps(z_comps_), func_vals(func_vals_), cubed_sphere_panels(cubed_sphere_panels_), degree(degree_), offset(offset_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double min_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).min_eta;
		double max_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).max_eta;
		double min_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).min_xi;
		double max_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).max_xi;
		double xi_derivs_workspace[121], eta_derivs_workspace[121];
		double xi_offset = 0.5*(min_xi + max_xi);
		double xi_scale = 0.5*(max_xi - min_xi);
		double eta_offset = 0.5*(min_eta + max_eta);
		double eta_scale = 0.5*(max_eta - min_eta);
		double xi, eta;
		bli_deriv_xi(xi_derivs_workspace, &func_vals(i-offset,0), degree, min_xi, max_xi);
		bli_deriv_eta(eta_derivs_workspace, &func_vals(i-offset, 0), degree, min_eta, max_eta);
		int index;
		for (int j = 0; j < degree+1; j++) { // xi loop
			xi = Kokkos::cos(Kokkos::numbers::pi * j / degree)* xi_scale + xi_offset;
			for (int k = 0; k < degree+1; k++) { // eta loop
				index = j * (degree+1) + k;
				eta = Kokkos::cos(Kokkos::numbers::pi * k / degree) * eta_scale + eta_offset;
				xyzvec_from_xietavec(x_comps(i-offset, index), y_comps(i-offset, index), z_comps(i-offset, index), xi_derivs_workspace[index], eta_derivs_workspace[index], cubed_sphere_panels(i).face, xi, eta);
			}
		}
	}
};

struct panel_gradient_xieta {
	Kokkos::View<double**, Kokkos::LayoutStride> xi_comps;
	Kokkos::View<double**, Kokkos::LayoutStride> eta_comps;
	Kokkos::View<double**, Kokkos::LayoutRight> func_vals;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int degree;
	int offset;

	panel_gradient_xieta(Kokkos::View<double**, Kokkos::LayoutStride>& xi_comps_, Kokkos::View<double**, Kokkos::LayoutStride>& eta_comps_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& func_vals_, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, 
					int degree_, int offset_) : xi_comps(xi_comps_), eta_comps(eta_comps_), func_vals(func_vals_), 
					cubed_sphere_panels(cubed_sphere_panels_), degree(degree_), offset(offset_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double min_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).min_eta;
		double max_eta = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).max_eta;
		double min_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).min_xi;
		double max_xi = Kokkos::numbers::pi / 4.0 * cubed_sphere_panels(i).max_xi;
		double xi_derivs[121], eta_derivs[121];
		double xi_offset = 0.5*(min_xi + max_xi);
		double xi_scale = 0.5*(max_xi - min_xi);
		double eta_offset = 0.5*(min_eta + max_eta);
		double eta_scale = 0.5*(max_eta - min_eta);
		double xi, eta;
		double X, Y, X2, Y2, C, D;
		bli_deriv_xi(xi_derivs, &func_vals(i-offset,0), degree, min_xi, max_xi);
		bli_deriv_eta(eta_derivs, &func_vals(i-offset, 0), degree, min_eta, max_eta);
		int index;
		for (int j = 0; j < degree+1; j++) { // xi loop
			xi = Kokkos::cos(Kokkos::numbers::pi * j / degree)* xi_scale + xi_offset;
			X = Kokkos::tan(xi);
			C = Kokkos::sqrt(1+X*X);
			for (int k = 0; k < degree+1; k++) { // eta loop
				index = j * (degree+1) + k;
				eta = Kokkos::cos(Kokkos::numbers::pi * k / degree) * eta_scale + eta_offset;
				Y = Kokkos::tan(eta);
				D = Kokkos::sqrt(1+Y*Y);
				xi_comps(i-offset,index) = D*xi_derivs[index] + X*Y/D*eta_derivs[index];
				eta_comps(i-offset,index) = X*Y/C*xi_derivs[index] + C*eta_derivs[index];
			}
		}
	}
};

KOKKOS_INLINE_FUNCTION
void single_panel_grad(double* x_comps, double* y_comps, double* z_comps, double* f_vals, int degree, double min_xi, double max_xi, double min_eta, double max_eta, int face) {
	double xi_offset = 0.5*(min_xi + max_xi);
	double xi_scale = 0.5*(max_xi - min_xi);
	double eta_offset = 0.5*(min_eta + max_eta);
	double eta_scale = 0.5*(max_eta - min_eta);
	double xi, eta, X, Y, C, D;
	double xi_derivs_workspace[121], eta_derivs_workspace[121], xi_comp[121], eta_comp[121];
	bli_deriv_xi(xi_derivs_workspace, f_vals, degree, min_xi, max_xi);
	bli_deriv_eta(eta_derivs_workspace, f_vals, degree, min_eta, max_eta);
	int index = 0;
	for (int i = 0; i < degree+1; i++) {
		xi = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_scale + xi_offset;
		for (int j = 0; j < degree+1; j++) {
			eta = Kokkos::cos(Kokkos::numbers::pi * j / degree) * eta_scale + eta_offset;
			X = Kokkos::tan(xi);
			Y = Kokkos::tan(eta);
			C = Kokkos::sqrt(1+X*X);
			D = Kokkos::sqrt(1+Y*Y);
			xi_comp[index] = D*xi_derivs_workspace[index] + X*Y/D*eta_derivs_workspace[index];
			eta_comp[index] = X*Y/C*xi_derivs_workspace[index] + C*eta_derivs_workspace[index];
			// xi_comp[index] /= 6371000.0;
			// eta_comp[index] /= 6371000.0;
			// xi_comp[index] = xi_derivs_workspace[index];
			// eta_comp[index] = eta_derivs_workspace[index];
			index += 1;
		}
	}
	index = 0;
	for (int i = 0; i < degree+1; i++) {
		xi = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_scale + xi_offset;
		for (int j = 0; j < degree+1; j++) {
			eta = Kokkos::cos(Kokkos::numbers::pi * j / degree) * eta_scale + eta_offset;
			xyzvec_from_xietavec(x_comps[index], y_comps[index], z_comps[index], xi_comp[index], eta_comp[index], face, xi, eta);
			index += 1;
		}
	}
}

KOKKOS_INLINE_FUNCTION
void single_panel_jac_xieta(double* xi_comps, double* eta_comps, double* jac_vals, int degree, double min_xi, double max_xi, double min_eta, double max_eta, int face) {
	double xi_offset = 0.5*(min_xi + max_xi);
	double xi_scale = 0.5*(max_xi - min_xi);
	double eta_offset = 0.5*(min_eta + max_eta);
	double eta_scale = 0.5*(max_eta - min_eta);
	double xi, eta;
	double xixi_derivs[121], xieta_derivs[121], etaxi_derivs[121], etaeta_derivs[121];
	bli_deriv_xi(xixi_derivs, xi_comps, degree, min_xi, max_xi);
	bli_deriv_eta(xieta_derivs, xi_comps, degree, min_eta, max_eta);
	bli_deriv_xi(etaxi_derivs, eta_comps, degree, min_xi, max_xi);
	bli_deriv_eta(etaeta_derivs, eta_comps, degree, min_eta, max_eta);
	int index;
	double X, Y, X2, Y2, C, D;
	for (int i = 0; i < degree+1; i++) {
		xi = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_scale + xi_offset;
		X = Kokkos::tan(xi);
		C = Kokkos::sqrt(1+X*X);
		for (int j = 0; j < degree+1; j++) {
			eta = Kokkos::cos(Kokkos::numbers::pi * j / degree) * eta_scale + eta_offset;
			Y = Kokkos::tan(eta);
			D = Kokkos::sqrt(1+Y*Y);
			index = i * (degree+1) + j;
			jac_vals[4*index] = D*xixi_derivs[index] + X*Y/D*xieta_derivs[index];
			jac_vals[4*index+1] = X*Y/C*xixi_derivs[index] + C*xieta_derivs[index];
			jac_vals[4*index+2] = D*etaxi_derivs[index] + X*Y/D*etaeta_derivs[index];
			jac_vals[4*index+3] = X*Y/C*etaxi_derivs[index] + C*etaeta_derivs[index];
		}
	}
}

struct panel_laplacian {
	Kokkos::View<double**, Kokkos::LayoutRight> laplacian_vals;
	Kokkos::View<double**, Kokkos::LayoutRight> func_vals;
	Kokkos::View<double**, Kokkos::LayoutRight> xi_derivs_workspace;
	Kokkos::View<double**, Kokkos::LayoutRight> eta_derivs_workspace;
	Kokkos::View<double**, Kokkos::LayoutRight> xieta_derivs_workspace;
	Kokkos::View<double**, Kokkos::LayoutRight> xixi_derivs_workspace;
	Kokkos::View<double**, Kokkos::LayoutRight> etaeta_derivs_workspace;
	Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
	int degree;
	int offset;

	panel_laplacian(Kokkos::View<double**, Kokkos::LayoutRight>& laplacian_vals_, Kokkos::View<double**, Kokkos::LayoutRight>& func_vals_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& xi_derivs_workspace_, Kokkos::View<double**, Kokkos::LayoutRight>& eta_derivs_workspace_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& xieta_derivs_workspace_, Kokkos::View<double**, Kokkos::LayoutRight>& xixi_derivs_workspace_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& etaeta_derivs_workspace_, Kokkos::View<CubedSpherePanel*>& cubed_sphere_panels_, int degree_, int offset_) : 
					laplacian_vals(laplacian_vals_), func_vals(func_vals_), xi_derivs_workspace(xi_derivs_workspace_), eta_derivs_workspace(eta_derivs_workspace_), xieta_derivs_workspace(xieta_derivs_workspace_), 
					xixi_derivs_workspace(xixi_derivs_workspace_), etaeta_derivs_workspace(etaeta_derivs_workspace_), cubed_sphere_panels(cubed_sphere_panels_), degree(degree_), offset(offset_) {}

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
		bli_deriv_xi(&xi_derivs_workspace(i-offset,0), &func_vals(i-offset,0), degree, min_xi, max_xi);
		bli_deriv_eta(&eta_derivs_workspace(i-offset,0), &func_vals(i-offset, 0), degree, min_eta, max_eta);
		bli_deriv_xi(&xixi_derivs_workspace(i-offset,0), &xi_derivs_workspace(i-offset,0), degree, min_xi, max_xi);
		bli_deriv_xi(&xieta_derivs_workspace(i-offset,0), &eta_derivs_workspace(i-offset,0), degree, min_xi, max_xi);
		bli_deriv_eta(&etaeta_derivs_workspace(i-offset,0), &eta_derivs_workspace(i-offset,0), degree, min_eta, max_eta);
		double xi, eta;
		int index;
		double X, Y, C2, D2, delta;
		for (int j = 0; j < degree+1; j++) { // xi loop
			xi = Kokkos::cos(Kokkos::numbers::pi * j / degree) * xi_scale + xi_offset;
			for (int k = 0; k < degree+1; k++) { // eta loop
				eta = Kokkos::cos(Kokkos::numbers::pi * k / degree) * eta_scale + eta_offset;
				index = j * (degree+1)+k;
				X = Kokkos::tan(xi);
				Y = Kokkos::tan(eta);
				C2 = 1+X*X;
				D2 = 1+Y*Y;
				delta = 1+X*X+Y*Y;
				laplacian_vals(i-offset,index) = delta*(xixi_derivs_workspace(i-offset,index)/C2+etaeta_derivs_workspace(i-offset,index)/D2+2.0*X*Y/(C2*D2)*xieta_derivs_workspace(i-offset,index));
			}
		}
	}
};

KOKKOS_INLINE_FUNCTION

void spatial_filter_vals(double* output_vals, double* func_vals, int degree) {
	double coeff = 0.0005;
	double xi, eta, xi_d_1, xi_d_2, eta_d_1, eta_d_2, f1, f2, f3, f4;
	bool modify_xi, modify_eta;
	int index, index2;
	for (int i = 0; i < degree+1; i++) { // xi loop
		xi = Kokkos::cos(Kokkos::numbers::pi * i / degree);
		for (int j = 0; j < degree+1; j++) { // eta loop
			eta = Kokkos::cos(Kokkos::numbers::pi * j / degree);
			index = i * (degree+1) + j;
			output_vals[index] = (1.0-coeff)*func_vals[index];
			if (i == 0) {
				xi_d_1 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i+1) / degree));
				xi_d_2 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i+2) / degree));
				f1 = func_vals[(i+1)*(degree+1)+j];
				f2 = func_vals[(i+2)*(degree+1)+j];
			} else if (i == degree) {
				xi_d_1 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i-1) / degree));
				xi_d_2 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i-2) / degree));
				f1 = func_vals[(i-1)*(degree+1)+j];
				f2 = func_vals[(i-2)*(degree+1)+j];
			} else {
				xi_d_1 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i-1) / degree));
				xi_d_2 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i+1) / degree));
				f1 = func_vals[(i-1)*(degree+1)+j];
				f2 = func_vals[(i+1)*(degree+1)+j];
			} 
			if (j == 0) {
				eta_d_1 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j+1) / degree));
				eta_d_2 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j+2) / degree));
				f3 = func_vals[index+1];
				f4 = func_vals[index+2];
			} else if (j == degree) {
				eta_d_1 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j-1) / degree));
				eta_d_2 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j-2) / degree));
				f3 = func_vals[index-1];
				f4 = func_vals[index-2];
			} else {
				eta_d_1 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j-1) / degree));
				eta_d_2 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j+1) / degree));
				f3 = func_vals[index-1];
				f4 = func_vals[index+1];
			}
			output_vals[index] += 0.25*coeff/(xi_d_1+xi_d_2)*xi_d_1*f2;
			output_vals[index] += 0.25*coeff/(xi_d_1+xi_d_2)*xi_d_2*f1;
			output_vals[index] += 0.25*coeff/(eta_d_1+eta_d_2)*eta_d_1*f4;
			output_vals[index] += 0.25*coeff/(eta_d_1+eta_d_2)*eta_d_2*f3;
			// if (i == 0) { // xi edge, leave xi unchanged
			// 	modify_xi = false;
			// } else if (i == degree) { // xi edge, leave xi unchanged
			// 	modify_xi = false;
			// } else { // xi interior
			// 	modify_xi = true;
			// }
			// if (j == 0) { // eta edge, leave eta unchanged
			// 	modify_eta = false;
			// } else if (j == degree) {
			// 	modify_eta = false;
			// } else {
			// 	modify_eta = true;
			// }
			// if (modify_xi or modify_eta) {
			// 	output_vals[index] = (1.0-coeff) * func_vals[index];
			// 	if (modify_xi and modify_eta) {
			// 		xi_d_1 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i-1) / degree));
			// 		xi_d_2 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i+1) / degree));
			// 		eta_d_1 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j-1) / degree));
			// 		eta_d_2 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j+1) / degree));
			// 		index2 = (i+1)*(degree+1)+j;
			// 		output_vals[index] += 0.25*coeff/(xi_d_1+xi_d_2)*xi_d_1*func_vals[index2];
			// 		index2 = (i-1)*(degree+1)+j;
			// 		output_vals[index] += 0.25*coeff/(xi_d_1+xi_d_2)*xi_d_2*func_vals[index2];
			// 		output_vals[index] += 0.25*coeff/(eta_d_1+eta_d_2)*eta_d_1*func_vals[index+1];
			// 		output_vals[index] += 0.25*coeff/(eta_d_1+eta_d_2)*eta_d_2*func_vals[index-1];
			// 	} else if (modify_xi) {
			// 		xi_d_1 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i-1) / degree));
			// 		xi_d_2 = Kokkos::abs(xi - Kokkos::cos(Kokkos::numbers::pi * (i+1) / degree));
			// 		index2 = (i+1)*(degree+1)+j;
			// 		output_vals[index] += 0.5*coeff/(xi_d_1+xi_d_2)*xi_d_1*func_vals[index2];
			// 		index2 = (i-1)*(degree+1)+j;
			// 		output_vals[index] += 0.5*coeff/(xi_d_1+xi_d_2)*xi_d_2*func_vals[index2];
			// 	} else if (modify_eta) {
			// 		eta_d_1 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j-1) / degree));
			// 		eta_d_2 = Kokkos::abs(eta - Kokkos::cos(Kokkos::numbers::pi * (j+1) / degree));
			// 		output_vals[index] += 0.5*coeff/(eta_d_1+eta_d_2)*eta_d_1*func_vals[index+1];
			// 		output_vals[index] += 0.5*coeff/(eta_d_1+eta_d_2)*eta_d_2*func_vals[index-1];
			// 	}
			// } else {
			// 	output_vals[index] = func_vals[index];
			// }
		}
	}
}

KOKKOS_INLINE_FUNCTION
void single_panel_lap(double* lap_vals, double* func_vals, double min_xi, double max_xi, double min_eta, double max_eta, int degree) {
	double xi_derivs[121], eta_derivs[121], xixi_derivs[121], xieta_derivs[121], etaeta_derivs[121], etaxi_derivs[121], filter_vals[121];
	spatial_filter_vals(filter_vals, func_vals, degree);
	bli_deriv_xi(xi_derivs, filter_vals, degree, min_xi, max_xi);
	bli_deriv_eta(eta_derivs, filter_vals, degree, min_eta, max_eta);
	bli_deriv_xi(xixi_derivs, xi_derivs, degree, min_xi, max_xi);
	bli_deriv_eta(etaeta_derivs, eta_derivs, degree, min_eta, max_eta);
	bli_deriv_eta(xieta_derivs, xi_derivs, degree, min_eta, max_eta);
	bli_deriv_xi(etaxi_derivs, eta_derivs, degree, min_xi, max_xi);
	double xi, eta;
	int index;
	double X, Y, C2, D2, delta;
	double xi_offset = 0.5*(min_xi + max_xi);
	double xi_scale = 0.5*(max_xi - min_xi);
	double eta_offset = 0.5*(min_eta + max_eta);
	double eta_scale = 0.5*(max_eta - min_eta);
	double mix_der;
	for (int i = 0; i < degree+1; i++) { // xi loop
		xi = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_scale + xi_offset;
		for (int j = 0; j < degree+1; j++) { // eta loop
			eta = Kokkos::cos(Kokkos::numbers::pi * j / degree) * eta_scale + eta_offset;
			index = i * (degree+1) + j;
			X = Kokkos::tan(xi);
			Y = Kokkos::tan(eta);
			C2 = 1+X*X;
			D2 = 1+Y*Y;
			delta = 1+X*X+Y*Y;
			mix_der = 0.5*(xieta_derivs[index] + etaxi_derivs[index]);
			lap_vals[index] = delta*(xixi_derivs[index]/C2 + etaeta_derivs[index]/D2 +2.0*X*Y/(C2*D2)*mix_der);
		}
	}
}

KOKKOS_INLINE_FUNCTION
void single_panel_lap_no_filter(double* lap_vals, double* func_vals, double min_xi, double max_xi, double min_eta, double max_eta, int degree) {
	double xi_derivs[121], eta_derivs[121], xixi_derivs[121], xieta_derivs[121], etaeta_derivs[121], etaxi_derivs[121], filter_vals[121];
	bli_deriv_xi(xi_derivs, func_vals, degree, min_xi, max_xi);
	bli_deriv_eta(eta_derivs, func_vals, degree, min_eta, max_eta);
	bli_deriv_xi(xixi_derivs, xi_derivs, degree, min_xi, max_xi);
	bli_deriv_eta(etaeta_derivs, eta_derivs, degree, min_eta, max_eta);
	bli_deriv_eta(xieta_derivs, xi_derivs, degree, min_eta, max_eta);
	bli_deriv_xi(etaxi_derivs, eta_derivs, degree, min_xi, max_xi);
	double xi, eta;
	int index;
	double X, Y, C2, D2, delta;
	double xi_offset = 0.5*(min_xi + max_xi);
	double xi_scale = 0.5*(max_xi - min_xi);
	double eta_offset = 0.5*(min_eta + max_eta);
	double eta_scale = 0.5*(max_eta - min_eta);
	double mix_der;
	for (int i = 0; i < degree+1; i++) { // xi loop
		xi = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_scale + xi_offset;
		for (int j = 0; j < degree+1; j++) { // eta loop
			eta = Kokkos::cos(Kokkos::numbers::pi * j / degree) * eta_scale + eta_offset;
			index = i * (degree+1) + j;
			X = Kokkos::tan(xi);
			Y = Kokkos::tan(eta);
			C2 = 1+X*X;
			D2 = 1+Y*Y;
			delta = 1+X*X+Y*Y;
			mix_der = 0.5*(xieta_derivs[index] + etaxi_derivs[index]);
			lap_vals[index] = delta*(xixi_derivs[index]/C2 + etaeta_derivs[index]/D2 +2.0*X*Y/(C2*D2)*mix_der);
		}
	}
}

KOKKOS_INLINE_FUNCTION
void single_panel_curl_rad_comp(double* curl_rad_vals, double* func_vals_xi, double* func_vals_eta, double min_xi, double max_xi, double min_eta, double max_eta, int degree) {
	double xi_derivs_xi[121], eta_derivs_xi[121], xi_derivs_eta[121], eta_derivs_eta[121], filter_vals_xi[121], filter_vals_eta[121];
	spatial_filter_vals(filter_vals_xi, func_vals_xi, degree);
	spatial_filter_vals(filter_vals_eta, func_vals_eta, degree);
	bli_deriv_xi(xi_derivs_xi, filter_vals_xi, degree, min_xi, max_xi);
	bli_deriv_xi(xi_derivs_eta, filter_vals_eta, degree, min_xi, max_xi);
	bli_deriv_eta(eta_derivs_xi, filter_vals_xi, degree, min_eta, max_eta);
	bli_deriv_eta(eta_derivs_eta, filter_vals_eta, degree, min_eta, max_eta);
	double xi, eta, X, Y, C, D, delta;
	int index = 0;
	double xi_offset = 0.5*(min_xi + max_xi);
	double xi_scale = 0.5*(max_xi - min_xi);
	double eta_offset = 0.5*(min_eta + max_eta);
	double eta_scale = 0.5*(max_eta - min_eta);
	for (int i = 0; i < degree+1; i++) {
		xi = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_scale + xi_offset;
		for (int j = 0; j < degree+1; j++) {
			eta = Kokkos::cos(Kokkos::numbers::pi * j / degree) * eta_scale + eta_offset;
			X = Kokkos::tan(xi);
			Y = Kokkos::tan(eta);
			C = Kokkos::sqrt(1+X*X);
			D = Kokkos::sqrt(1+Y*Y);
			delta = 1+X*X+Y*Y;
			curl_rad_vals[index] = Kokkos::sqrt(delta)* (X*Y/(C*D)*(eta_derivs_eta[index]/D-xi_derivs_xi[index]/C)-eta_derivs_xi[index]/D+xi_derivs_eta[index]/C);
			index += 1;
		}
	}
}

KOKKOS_INLINE_FUNCTION
void single_panel_div(double* div_vals, double* func_vals_xi, double* func_vals_eta, double min_xi, double max_xi, double min_eta, double max_eta, int degree) {
	double xi_derivs[121], eta_derivs[121], xi_scaled_vals[121], eta_scaled_vals[121], filter_vals_xi[121], filter_vals_eta[121];
	double X, Y, C, D, delta, xi, eta;
	int index = 0;
	double xi_offset = 0.5*(min_xi + max_xi);
	double xi_scale = 0.5*(max_xi - min_xi);
	double eta_offset = 0.5*(min_eta + max_eta);
	double eta_scale = 0.5*(max_eta - min_eta);
	spatial_filter_vals(filter_vals_xi, func_vals_xi, degree);
	spatial_filter_vals(filter_vals_eta, func_vals_eta, degree);
	for (int i = 0; i < degree+1; i++) {
		xi = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_scale + xi_offset;
		for (int j = 0; j < degree+1; j++) {
			eta = Kokkos::cos(Kokkos::numbers::pi * j / degree) * eta_scale + eta_offset;
			X = Kokkos::tan(xi);
			Y = Kokkos::tan(eta);
			delta = 1+X*X+Y*Y;
			xi_scaled_vals[index] = filter_vals_xi[index] / Kokkos::sqrt(delta);
			eta_scaled_vals[index] = filter_vals_eta[index] / Kokkos::sqrt(delta);
			index += 1;
		}
	}
	bli_deriv_xi(xi_derivs, xi_scaled_vals, degree, min_xi, max_xi);
	bli_deriv_eta(eta_derivs, eta_scaled_vals, degree, min_eta, max_eta);
	for (int i = 0; i < degree+1; i++) {
		xi = Kokkos::cos(Kokkos::numbers::pi * i / degree) * xi_scale + xi_offset;
		for (int j = 0; j < degree+1; j++) {
			eta = Kokkos::cos(Kokkos::numbers::pi * j / degree) * eta_scale + eta_offset;
			X = Kokkos::tan(xi);
			Y = Kokkos::tan(eta);
			C = Kokkos::sqrt(1+X*X);
			D = Kokkos::sqrt(1+Y*Y);
			delta = 1+X*X+Y*Y;
			div_vals[index]=Kokkos::pow(delta, 1.5)/(C*D) * (xi_derivs[index]/C+eta_derivs[index]/D);
			index += 1;
		}
	}
}

#endif
