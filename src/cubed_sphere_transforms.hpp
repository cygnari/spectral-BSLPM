#ifndef H_CUBE_SPHERE_TRANSFORMS_H
#define H_CUBE_SPHERE_TRANSFORMS_H

double cubed_sphere_jac(double xi, double eta);

int face_from_xyz(const double x, const double y, const double z);

void xyz_from_xieta(const double xi, const double eta, const int face, double* xyz);

void xieta_from_xyz(const double x, const double y, const double z, double* xieta);

void xieta_from_xyz(const double x, const double y, const double z, const int face, double* xieta);

void loncolatvec_from_xietavec(double& lon_comp, double& colat_comp, const double xi_deriv, const double eta_deriv, const int face, const double xi, const double eta);

void xyzvec_from_xietavec(double& x_comp, double& y_comp, double& z_comp, const double xi_deriv, const double eta_deriv, const int face, const double xi, const double eta);

#endif
