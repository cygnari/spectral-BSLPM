#include <cmath>
#include <string>

double sphere_tri_area(const double* p1, const double* p2, const double* p3) {
	double p1n[3], p2n[3], p3n[3];
	double a, b, c, s, z, area;
	memcpy(p1n, p1, 3 * sizeof(double));
	memcpy(p2n, p2, 3 * sizeof(double));
	memcpy(p3n, p3, 3 * sizeof(double));
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