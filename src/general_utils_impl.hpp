#ifndef H_GENERAL_UTIL_IMPL_H
#define H_GENERAL_UTIL_IMPL_H

#include "Kokkos_Core.hpp"

KOKKOS_INLINE_FUNCTION
double sphere_tri_area_device(const double* p1, const double* p2, const double* p3) {
	double p1n[3], p2n[3], p3n[3];
	double a, b, c, s, z, area;
	p1n[0] = p1[0], p1n[1]=p1[1], p1n[2]=p1[2];
	p2n[0] = p2[0], p2n[1]=p2[1], p2n[2]=p2[2];
	p3n[0] = p3[0], p3n[1]=p3[1], p3n[2]=p3[2];
	p2n[0] -= p3[0], p2n[1] -= p3[1], p2n[2] -= p3[2];
	p3n[0] -= p1[0], p3n[1] -= p1[1], p3n[2] -= p1[2];
	p1n[0] -= p2[0], p1n[1] -= p2[1], p1n[2] -= p2[2];
	a = Kokkos::acos(1 - 0.5 * (p2n[0]*p2n[0]+p2n[1]*p2n[1]+p2n[2]*p2n[2]));
	b = Kokkos::acos(1 - 0.5 * (p3n[0]*p3n[0]+p3n[1]*p3n[1]+p3n[2]*p3n[2]));
	c = Kokkos::acos(1 - 0.5 * (p1n[0]*p1n[0]+p1n[1]*p1n[1]+p1n[2]*p1n[2]));
	s = (a + b + c) / 2;
	z = Kokkos::tan(s / 2) * Kokkos::tan((s - a) / 2) * Kokkos::tan((s - b) / 2) * Kokkos::tan((s - c) / 2);
	area = 4 * Kokkos::atan(Kokkos::sqrt(z));
	return area;
}

KOKKOS_INLINE_FUNCTION 
double gcdist(const double* p1, const double* p2) {
	double dp = Kokkos::fmax(Kokkos::fmin(p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2], 1.0), -1.0);
	return Kokkos::acos(dp);
}

KOKKOS_INLINE_FUNCTION 
double dilog(const double x) {
	// adapted from https://github.com/Expander/polylogarithm by Alexander Voigt
	const double pi2 = Kokkos::pow(Kokkos::numbers::pi, 2);
	const double P[] = {
	  0.9999999999999999502e+0,
	  -2.6883926818565423430e+0,
	  2.6477222699473109692e+0,
	  -1.1538559607887416355e+0,
	  2.0886077795020607837e-1,
	  -1.0859777134152463084e-2
	};
	const double Q[] = {
	  1.0000000000000000000e+0,
	  -2.9383926818565635485e+0,
	  3.2712093293018635389e+0,
	  -1.7076702173954289421e+0,
	  4.1596017228400603836e-1,
	  -3.9801343754084482956e-2,
	  8.2743668974466659035e-4
	};

	double y = 0, r = 0, s = 1;

	// transform to [0, 0.5] assuming that x is between 0 and 1
	if (x <= 0) {
	 return 0;
	} else if (x < 0.5) {
	 y = x;
	 r = 0;
	 s = 1;
	} else if (x == 1) {
	 return pi2 / 12.0 - 0.5*Kokkos::pow(Kokkos::log(2), 2);
	} else if (x < 1) {
	 y = 1 - x;
	 r = pi2/6.0 - Kokkos::log(x)*Kokkos::log(y);
	 s = -1;
	} else {
	 return pi2 / 6.0;
	}

	const double y2 = y*y;
	const double y4 = y2*y2;
	const double p = P[0] + y * P[1] + y2 * (P[2] + y * P[3]) +
	                y4 * (P[4] + y * P[5]);
	const double q = Q[0] + y * Q[1] + y2 * (Q[2] + y * Q[3]) +
	                y4 * (Q[4] + y * Q[5] + y2 * Q[6]);

	return r + s*y*p/q;
}

KOKKOS_INLINE_FUNCTION 
void xyzvec_from_loncolatvec(double& x_comp, double& y_comp, double& z_comp, const double lon_comp, const double colat_comp, const double x, const double y, const double z) {
	double sqc = Kokkos::sqrt(x*x+y*y);
	x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
	y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
	z_comp = -sqc*colat_comp;
}

KOKKOS_INLINE_FUNCTION 
void xyzvec_from_loncolatvec(double& x_comp, double& y_comp, double& z_comp, const double lon_comp, const double colat_comp, const double lon, const double colat) {
	double coslon = Kokkos::cos(lon);
	double sincolat = Kokkos::sin(colat);
	double coscolat = Kokkos::cos(colat);
	x_comp = coslon * coscolat * colat_comp - sincolat * lon_comp;
	y_comp = coslon * sincolat * colat_comp + coscolat * lon_comp;
	z_comp = Kokkos::sin(lon) * colat_comp; 
}

KOKKOS_INLINE_FUNCTION 
void loncolatvec_from_xyzvec(double& lon_comp, double& colat_comp, const double x_comp, const double y_comp, const double z_comp, const double x, const double y, const double z) {
	double sqc = Kokkos::sqrt(x*x+y*y);
	lon_comp = 1.0/sqc * (-y*x_comp + x*y_comp);
	colat_comp = z/sqc*(x*x_comp + y*y_comp) - sqc*z_comp;
}

struct copy_kokkos_view_2 {
	Kokkos::View<double**, Kokkos::LayoutRight> target;
	Kokkos::View<double**, Kokkos::LayoutRight> source;

	copy_kokkos_view_2(Kokkos::View<double**, Kokkos::LayoutRight>& target_, Kokkos::View<double**, Kokkos::LayoutRight>& source_) :
						target(target_), source(source_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		target(i,j) = source(i,j);
	}
};

struct copy_kokkos_view_3 {
	Kokkos::View<double***, Kokkos::LayoutRight> target;
	Kokkos::View<double***, Kokkos::LayoutRight> source;

	copy_kokkos_view_3(Kokkos::View<double***, Kokkos::LayoutRight>& target_, Kokkos::View<double***, Kokkos::LayoutRight>& source_) :
						target(target_), source(source_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j, const int k) const {
		target(i,j,k) = source(i,j,k);
	}
};

struct zero_out {
	Kokkos::View<double**, Kokkos::LayoutRight> vec;

	zero_out(Kokkos::View<double**, Kokkos::LayoutRight>& vec_) : vec(vec_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		vec(i,j) = 0;
	}
};

struct zero_out_1 {
	Kokkos::View<double*> vec;

	zero_out_1(Kokkos::View<double*>& vec_) : vec(vec_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		vec(i) = 0;
	}
};

KOKKOS_INLINE_FUNCTION
void project_to_sphere(double& x, double& y, double& z) {
	double pointnorm = Kokkos::sqrt(x*x+y*y+z*z);
	x /= pointnorm;
	y /= pointnorm;
	z /= pointnorm;
}

KOKKOS_INLINE_FUNCTION
void make_tangent_to_sphere(double x, double y, double z, double* vec) {
	double dp = x * vec[0] + y * vec[1] + z * vec[2];
	vec[0] -= dp * x;
	vec[1] -= dp * y;
	vec[2] -= dp * z;
}

struct xyz_vel_to_uv_vel {
	Kokkos::View<double**, Kokkos::LayoutRight> xcos;
	Kokkos::View<double**, Kokkos::LayoutRight> ycos;
	Kokkos::View<double**, Kokkos::LayoutRight> zcos;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_x;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_y;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_z;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_u;
	Kokkos::View<double**, Kokkos::LayoutRight> vel_v;

	xyz_vel_to_uv_vel(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& vel_u_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_v_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), 
					vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), vel_u(vel_u_), vel_v(vel_v_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		double x, y, z, xc, yc, zc;
		for (int j = 0; j < xcos.extent_int(1); j++) {
			x = xcos(i,j);
			y = ycos(i,j);
			z = zcos(i,j);
			xc = vel_x(i,j);
			yc = vel_y(i,j);
			zc = vel_z(i,j);
			if (abs(z) < 1 - 1e-16) { // away from pole
				vel_u(i,j) = (-y*xc + x*yc)/Kokkos::sqrt(x*x+y*y);
				vel_v(i,j) = ((x*xc+y*yc)*z-(x*x+y*y)*zc)/Kokkos::sqrt(x*x+y*y);
			} else {
				vel_u(i,j) = 0;
				vel_v(i,j) = 0;
			}
		}
	}
};

KOKKOS_INLINE_FUNCTION
void xyz_to_latlon(double& lat, double& lon, const double x, const double y, const double z) {
	// turns cartesian coordinates to spherical coordinates
	lat = M_PI / 2.0 - Kokkos::atan2(Kokkos::sqrt(x * x + y * y), z); // colatitude
	lon = Kokkos::atan2(y, x);                   							// longitude
}

struct filter_vals {
	Kokkos::View<double**, Kokkos::LayoutRight> vals;
	double thresh;

	filter_vals(Kokkos::View<double**, Kokkos::LayoutRight>& vals_, double thresh_) : vals(vals_), thresh(thresh_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int jmax = vals.extent_int(1);
		for (int j = 0; j < jmax; j++) {
			if (Kokkos::abs(vals(i,j)) < thresh) {
				vals(i,j) = 0.0;
			}
		}
	}
};

#endif
