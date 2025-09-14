#ifndef H_CUBE_SPHERE_TRANSFORMS_H
#define H_CUBE_SPHERE_TRANSFORMS_H

int face_from_xyz(const double x, const double y, const double z);

void xyz_from_xieta(const double xi, const double eta, const int face, double* xyz);

void xieta_from_xyz(const double x, const double y, const double z, double* xieta);

void xieta_from_xyz(const double x, const double y, const double z, const int face, double* xieta);

#endif
