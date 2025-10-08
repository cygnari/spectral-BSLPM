#include <Kokkos_Core.hpp>
#include <vector>
#include <cmath>

#include "run_config.hpp"

double bli_coeff(const int j, const int degree) {
  // bli weight
	if (degree == 0) {
		return 1;
	} else if (j == 0) {
		return 0.5;
	} else if (j == degree) {
		return 0.5 * pow(-1, j);
	} else {
		return pow(-1, j);
	}
}

void interp_init(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_info) {
	int interp_point_count = pow(run_config.interp_degree + 1, 2);
	int points_per_side = run_config.interp_degree + 1;

	std::vector<double> cheb_points (run_config.interp_degree+1, 0), bli_weights (run_config.interp_degree+1, 0);
	if (run_config.interp_degree == 0) {
		cheb_points[0] = 0;
		bli_weights[0] = 1;
	} else {
		for (int i = 0; i < run_config.interp_degree+1; i++) {
			cheb_points[i] = cos(M_PI / run_config.interp_degree * i);
			if (std::abs(cheb_points[i]) < 1e-16) {
				cheb_points[i] = 0;
			}
			bli_weights[i] = bli_coeff(i, run_config.interp_degree);
		}
	}

	std::vector<double> cc_weights (run_config.interp_degree+1, 0), vec (run_config.interp_degree / 2+1, 2);
	vec[0] = 1;
	for (int i = 1; i < run_config.interp_degree/2+1; i++) {
		vec[i] = 2.0 / (1 - 4 * i * i);
	}

	double sum;
	for (int i = 0; i < run_config.interp_degree/2+1; i++) {
		sum = 0;
		for (int j = 0; j < run_config.interp_degree/2+1; j++) {
			sum += 2*cos(2 * i * j * M_PI / run_config.interp_degree)/run_config.interp_degree * vec[j];
		}
		cc_weights[i] = sum;
	}
	cc_weights[0] *= 0.5;

	for (int i = 0; i < run_config.interp_degree/2; i++) {
		cc_weights[run_config.interp_degree-i] = cc_weights[i];
	}

	// interp_info[i][0] is xi value, [1] is eta value, [2] is bli coeff, [3] is C-C weight
	int index;
	for (int i = 0; i < points_per_side; i++) { // xi loop
		for (int j = 0; j < points_per_side; j++) { // eta loop
			index = i * points_per_side + j;
			interp_info(index,0) = cheb_points[i];
			interp_info(index,1) = cheb_points[j];
			interp_info(index,2) = bli_weights[i] * bli_weights[j];
			interp_info(index,3) = cc_weights[i] * cc_weights[j];
		}
	}
}

void interp_vals_bli(double* basis_vals, double xi, double eta, double min_xi, double max_xi, 
						double min_eta, double max_eta, int interp_deg) {
	double cheb_xi[interp_deg+1], cheb_eta[interp_deg+1], bli_weights[interp_deg+1], xi_range, xi_offset, eta_range, eta_offset, val;
	xi_range = 0.5*(max_xi - min_xi);
	xi_offset = 0.5*(max_xi + min_xi);
	eta_range = 0.5*(max_eta - min_eta);
	eta_offset = 0.5*(max_eta + min_eta);

	if (interp_deg == 0) {
		cheb_xi[0] = xi_offset;
		cheb_eta[0] = eta_offset;
		bli_weights[0] = 1;
	} else {
		for (int i = 0; i < interp_deg+1; i++) {
			cheb_xi[i] = cos(M_PI * i / interp_deg) * xi_range + xi_offset;
			cheb_eta[i] = cos(M_PI * i / interp_deg) * eta_range + eta_offset;
			if (i == 0) {
				bli_weights[i] = 0.5;
			} else if (i == interp_deg) {
				bli_weights[i] = 0.5 * pow(-1, i);
			} else {
				bli_weights[i] = pow(-1, i);
			}
		}
	}

	bool found_xi_point = false;
	double xi_f_vals[interp_deg+1], denom_xi;
	for (int i = 0; i < interp_deg+1; i++) {
		if (Kokkos::abs(xi - cheb_xi[i]) < 1e-16) {
			found_xi_point = true;
			for (int j = 0; j < interp_deg+1; j++) {
				xi_f_vals[j] = 0;
			}
			xi_f_vals[i] = 1;
			break;
		}
	}
	if (not found_xi_point) {
		denom_xi = 0;
		for (int i = 0; i < interp_deg+1; i++) {
			val = bli_weights[i] / (xi - cheb_xi[i]);
			xi_f_vals[i] = val;
			denom_xi += val;
		}
		for (int i = 0; i < interp_deg+1; i++) {
			xi_f_vals[i] /= denom_xi;
		}
	}

	bool found_eta_point = false;
	double eta_f_vals[interp_deg+1], denom_eta;
	for (int i = 0; i < interp_deg+1; i++) {
		if (Kokkos::abs(eta - cheb_eta[i]) < 1e-16) {
			found_eta_point = true;
			for (int j = 0; j < interp_deg+1; j++) {
				eta_f_vals[j] = 0;
			}
			eta_f_vals[i] = 1;
			break;
		}
	}
	if (not found_eta_point) {
		denom_eta = 0;
		for (int i = 0; i < interp_deg+1; i++) {
			val = bli_weights[i] / (eta - cheb_eta[i]);
			eta_f_vals[i] = val;
			denom_eta += val;
		}
		for (int i = 0; i < interp_deg+1; i++) {
			eta_f_vals[i] /= denom_eta;
		}
	}

	int index;
	for (int i = 0; i < interp_deg+1; i++) {
		for (int j = 0; j < interp_deg+1; j++) {
			index = i * (interp_deg+1) + j;
			basis_vals[index] = xi_f_vals[i] * eta_f_vals[j];
		}
	}
}