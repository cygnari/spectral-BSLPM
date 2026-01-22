#include <cmath>
#include <cstring>
#include <algorithm>

double sphere_tri_area(const double* p1, const double* p2, const double* p3) {
	double p1n[3], p2n[3], p3n[3];
	double a, b, c, s, z, area;
	std::memcpy(p1n, p1, 3 * sizeof(double));
	std::memcpy(p2n, p2, 3 * sizeof(double));
	std::memcpy(p3n, p3, 3 * sizeof(double));
	p2n[0] -= p3[0], p2n[1] -= p3[1], p2n[2] -= p3[2];
	p3n[0] -= p1[0], p3n[1] -= p1[1], p3n[2] -= p1[2];
	p1n[0] -= p2[0], p1n[1] -= p2[1], p1n[2] -= p2[2];
	a = acos(1 - 0.5 * (p2n[0]*p2n[0]+p2n[1]*p2n[1]+p2n[2]*p2n[2]));
	b = acos(1 - 0.5 * (p3n[0]*p3n[0]+p3n[1]*p3n[1]+p3n[2]*p3n[2]));
	c = acos(1 - 0.5 * (p1n[0]*p1n[0]+p1n[1]*p1n[1]+p1n[2]*p1n[2]));
	s = (a + b + c) / 2;
	z = tan(s / 2) * tan((s - a) / 2) * tan((s - b) / 2) * tan((s - c) / 2);
	area = 4 * atan(sqrt(z));
	return area;
}

double gcdist(const double* p1, const double* p2) {
	double dp = std::max(std::min(p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2], 1.0), -1.0);
	return acos(dp);
}

void xyz_to_colatlon(double& colat, double& lon, const double x, const double y, const double z) {
  // turns cartesian coordinates to spherical coordinates
	colat = atan2(sqrt(x * x + y * y), z); // colatitude
	lon = atan2(y, x);                     // longitude
}

void xyz_to_latlon(double& lat, double& lon, const double x, const double y, const double z) {
  // turns cartesian coordinates to spherical coordinates
	lat = M_PI / 2.0 - atan2(sqrt(x * x + y * y), z); // colatitude
	lon = atan2(y, x);                   							// longitude
}

void xyzvec_from_loncolatvec(double& x_comp, double& y_comp, double& z_comp, const double lon_comp, const double colat_comp, const double x, const double y, const double z) {
	double sqc = sqrt(x*x+y*y);
	x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
	y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
	z_comp = -sqc*colat_comp;
}

void xyzvec_from_loncolatvec(double& x_comp, double& y_comp, double& z_comp, const double lon_comp, const double colat_comp, const double lon, const double colat) {
	double coslon = cos(lon);
	double sincolat = sin(colat);
	double coscolat = cos(colat);
	x_comp = coslon * coscolat * colat_comp - sincolat * lon_comp;
	y_comp = coslon * sincolat * colat_comp + coscolat * lon_comp;
	z_comp = sin(lon) * colat_comp; 
}

void loncolatvec_from_xyzvec(double& lon_comp, double& colat_comp, const double x_comp, const double y_comp, const double z_comp, const double x, const double y, const double z) {
	double sqc = sqrt(x*x+y*y);
	lon_comp = 1.0/sqc * (-y*x_comp + x*y_comp);
	colat_comp = z/sqc*(x*x_comp + y*y_comp) - sqc*z_comp;
}

void xyz_from_lonlat(double& x_comp, double& y_comp, double& z_comp, const double lon, const double lat) {
	x_comp = cos(lat) * cos(lon);
	y_comp = cos(lat) * sin(lon);
	z_comp = sin(lat);
}