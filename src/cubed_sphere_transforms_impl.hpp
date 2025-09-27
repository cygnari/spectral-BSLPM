#ifndef H_CUBE_SPHERE_TRANSFORMS_IMPL_H
#define H_CUBE_SPHERE_TRANSFORMS_IMPL_H

#include <Kokkos_Core.hpp>


inline int face_from_xyz(const double x, const double y, const double z) {
	double ax = Kokkos::abs(x);
	double ay = Kokkos::abs(y);
	double az = Kokkos::abs(z);
	if ((ax >= ay) and (ax >= az)) {
	  if (x >= 0) {
	    return 1;
	  } else {
	    return 3;
	  }
	} else if ((ay >= ax) and (ay >= az)) {
	  if (y >= 0) {
	    return 2;
	  } else {
	    return 4;
	  }
	} else {
	  if (z >= 0) {
	    return 5;
	  } else {
	    return 6;
	  }
	}
}

inline void xyz_from_xieta_1(const double xi, const double eta, double* xyz) {
  double X = Kokkos::tan(xi);
  double Y = Kokkos::tan(eta);
  xyz[0] = 1.0/Kokkos::sqrt(1+X*X+Y*Y);
  xyz[1] = X*xyz[0];
  xyz[2] = Y*xyz[0];
}

inline void xyz_from_xieta_2(const double xi, const double eta, double* xyz) {
  double X = Kokkos::tan(xi);
  double Y = Kokkos::tan(eta);
  xyz[1] = 1.0/Kokkos::sqrt(1+X*X+Y*Y);
  xyz[0] = -X*xyz[1];
  xyz[2] = Y*xyz[1];
}

inline void xyz_from_xieta_3(const double xi, const double eta, double* xyz) {
  double X = Kokkos::tan(xi);
  double Y = Kokkos::tan(eta);
  xyz[0] = -1.0/Kokkos::sqrt(1+X*X+Y*Y);
  xyz[1] = X*xyz[0];
  xyz[2] = -Y*xyz[0];
}

inline void xyz_from_xieta_4(const double xi, const double eta, double* xyz) {
  double X = Kokkos::tan(xi);
  double Y = Kokkos::tan(eta);
  xyz[1] = -1.0/Kokkos::sqrt(1+X*X+Y*Y);
  xyz[0] = -X*xyz[1];
  xyz[2] = -Y*xyz[1];
}

inline void xyz_from_xieta_5(const double xi, const double eta, double* xyz) {
  double X = Kokkos::tan(xi);
  double Y = Kokkos::tan(eta);
  xyz[2] = 1.0/Kokkos::sqrt(1+X*X+Y*Y);
  xyz[0] = -Y*xyz[2];
  xyz[1] = X*xyz[2];
}

inline void xyz_from_xieta_6(const double xi, const double eta, double* xyz) {
  double X = Kokkos::tan(xi);
  double Y = Kokkos::tan(eta);
  xyz[2] = -1.0/Kokkos::sqrt(1+X*X+Y*Y);
  xyz[0] = -Y*xyz[2];
  xyz[1] = -X*xyz[2];
}

inline void xyz_from_xieta(const double xi, const double eta, const int face, double* xyz) {
	if (face == 1) {
	  xyz_from_xieta_1(xi, eta, xyz);
	} else if (face == 2) {
	  xyz_from_xieta_2(xi, eta, xyz);
	} else if (face == 3) {
	  xyz_from_xieta_3(xi, eta, xyz);
	} else if (face == 4) {
	  xyz_from_xieta_4(xi, eta, xyz);
	} else if (face == 5) {
	  xyz_from_xieta_5(xi, eta, xyz);
	} else if (face == 6) {
	  xyz_from_xieta_6(xi, eta, xyz);
	} else {
	  Kokkos::abort("Input face is not between 1 and 6, xyz_from_xieta");
	}
}

// void xieta_from_xyz_1(const double x, const double y, const double z, double* xieta) {
//   xieta[0] = Kokkos::atan(y/x);
//   xieta[1] = Kokkos::atan(z/x);
// }

// void xieta_from_xyz_2(const double x, const double y, const double z, double* xieta) {
//   xieta[0] = Kokkos::atan(-x/y);
//   xieta[1] = Kokkos::atan(z/y);
// }

// void xieta_from_xyz_3(const double x, const double y, const double z, double* xieta) {
//   xieta[0] = Kokkos::atan(y/x);
//   xieta[1] = Kokkos::atan(-z/x);
// }

// void xieta_from_xyz_4(const double x, const double y, const double z, double* xieta) {
//   xieta[0] = Kokkos::atan(-x/y);
//   xieta[1] = Kokkos::atan(-z/y);
// }

// void xieta_from_xyz_5(const double x, const double y, const double z, double* xieta) {
//   xieta[0] = Kokkos::atan(y/z);
//   xieta[1] = Kokkos::atan(-x/z);
// }

// void xieta_from_xyz_6(const double x, const double y, const double z, double* xieta) {
//   xieta[0] = Kokkos::atan(-y/z);
//   xieta[1] = Kokkos::atan(-x/z);
// }

inline void xieta_from_xyz(const double x, const double y, const double z, double* xieta) {
	double ax = Kokkos::abs(x);
	double ay = Kokkos::abs(y);
	double az = Kokkos::abs(z);

	if ((ax >= ay) and (ax >= az)) {
	  if (x >= 0) {
	    // xieta_from_xyz_1(x, y, z, xieta);
	    xieta[0] = Kokkos::atan(y/x);
	    xieta[1] = Kokkos::atan(z/x);
	  } else {
	    // xieta_from_xyz_3(x, y, z, xieta);
	    xieta[0] = Kokkos::atan(y/x);
      xieta[1] = Kokkos::atan(-z/x);
	  }
	} else if ((ay >= ax) and (ay >= az)) {
	  if (y >= 0) {
	    // xieta_from_xyz_2(x, y, z, xieta);
	    xieta[0] = Kokkos::atan(-x/y);
      xieta[1] = Kokkos::atan(z/y);
	  } else {
	    // xieta_from_xyz_4(x, y, z, xieta);
	    xieta[0] = Kokkos::atan(-x/y);
	    xieta[1] = Kokkos::atan(-z/y);
	  }
	} else {
	  if (z >= 0) {
	    // xieta_from_xyz_5(x, y, z, xieta);
	    xieta[0] = Kokkos::atan(y/z);
	    xieta[1] = Kokkos::atan(-x/z);
	  } else {
	    // xieta_from_xyz_6(x, y, z, xieta);
	    xieta[0] = Kokkos::atan(-y/z);
	    xieta[1] = Kokkos::atan(-x/z);
	  }
	}
}

inline void xieta_from_xyz(const double x, const double y, const double z, const int face, double* xieta) {
	if (face == 1) {
	  // xieta_from_xyz_1(x, y, z, xieta);
	  xieta[0] = Kokkos::atan(y/x);
    xieta[1] = Kokkos::atan(z/x);
	} else if (face == 2) {
	  // xieta_from_xyz_2(x, y, z, xieta);
    xieta[0] = Kokkos::atan(-x/y);
    xieta[1] = Kokkos::atan(z/y);
	} else if (face == 3) {
	  // xieta_from_xyz_3(x, y, z, xieta);
    xieta[0] = Kokkos::atan(y/x);
    xieta[1] = Kokkos::atan(-z/x);
	} else if (face == 4) {
	  // xieta_from_xyz_4(x, y, z, xieta);
	  xieta[0] = Kokkos::atan(-x/y);
	  xieta[1] = Kokkos::atan(-z/y);
	} else if (face == 5) {
	  // xieta_from_xyz_5(x, y, z, xieta);
	  xieta[0] = Kokkos::atan(y/z);
	  xieta[1] = Kokkos::atan(-x/z);
	} else if (face == 6) {
	  // xieta_from_xyz_6(x, y, z, xieta);
	  xieta[0] = Kokkos::atan(-y/z);
	  xieta[1] = Kokkos::atan(-x/z);
	} else {
	  Kokkos::abort("Input face is not between 1 and 6, xieta_from_xyz");
	}
}

#endif