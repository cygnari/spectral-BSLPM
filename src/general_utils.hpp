#ifndef H_GENERAL_UTIL_H
#define H_GENERAL_UTIL_H

double sphere_tri_area(const double* p1, const double* p2, const double* p3);

double gcdist(const double* p1, const double* p2);

void xyz_to_colatlon(double& colat, double& lon, const double x, const double y, const double z);

void xyz_to_latlon(double& lat, double& lon, const double x, const double y, const double z);

void xyzvec_from_loncolatvec(double& x_comp, double& y_comp, double& z_comp, const double lon_comp, const double colat_comp, const double x, const double y, const double z);

void xyzvec_from_loncolatvec(double& x_comp, double& y_comp, double& z_comp, const double lon_comp, const double colat_comp, const double lon, const double colat);

void loncolatvec_from_xyzvec(double& lon_comp, double& colat_comp, const double x_comp, const double y_comp, const double z_comp, const double x, const double y, const double z);

#endif
