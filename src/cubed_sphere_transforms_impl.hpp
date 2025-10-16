#ifndef H_CUBE_SPHERE_TRANSFORMS_IMPL_H
#define H_CUBE_SPHERE_TRANSFORMS_IMPL_H

#include <Kokkos_Core.hpp>
// #include <KokkosLapack_gesv.hpp>

KOKKOS_INLINE_FUNCTION
int face_from_xyz(const double x, const double y, const double z) {
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

KOKKOS_INLINE_FUNCTION
void xyz_from_xieta(const double xi, const double eta, const int face, double* xyz) {
	double X = Kokkos::tan(xi);
	double Y = Kokkos::tan(eta);
	if (face == 1) {
	  xyz[0] = 1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[1] = X*xyz[0];
	  xyz[2] = Y*xyz[0];
	} else if (face == 2) {
	  xyz[1] = 1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[0] = -X*xyz[1];
	  xyz[2] = Y*xyz[1];
	} else if (face == 3) {
	  xyz[0] = -1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[1] = X*xyz[0];
	  xyz[2] = -Y*xyz[0];
	} else if (face == 4) {
	  xyz[1] = -1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[0] = -X*xyz[1];
	  xyz[2] = -Y*xyz[1];
	} else if (face == 5) {
	  xyz[2] = 1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[0] = -Y*xyz[2];
	  xyz[1] = X*xyz[2];
	} else if (face == 6) {
	  xyz[2] = -1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[0] = -Y*xyz[2];
	  xyz[1] = -X*xyz[2];
	} else {
	  Kokkos::abort("Input face is not between 1 and 6, xyz_from_xieta");
	}
}

KOKKOS_INLINE_FUNCTION
void xieta_from_xyz(const double x, const double y, const double z, double* xieta) {
	double ax = Kokkos::abs(x);
	double ay = Kokkos::abs(y);
	double az = Kokkos::abs(z);

	if ((ax >= ay) and (ax >= az)) {
	  if (x >= 0) {
	    xieta[0] = Kokkos::atan(y/x);
	    xieta[1] = Kokkos::atan(z/x);
	  } else {
	    xieta[0] = Kokkos::atan(y/x);
      xieta[1] = Kokkos::atan(-z/x);
	  }
	} else if ((ay >= ax) and (ay >= az)) {
	  if (y >= 0) {
	    xieta[0] = Kokkos::atan(-x/y);
      xieta[1] = Kokkos::atan(z/y);
	  } else {
	    xieta[0] = Kokkos::atan(-x/y);
	    xieta[1] = Kokkos::atan(-z/y);
	  }
	} else {
	  if (z >= 0) {
	    xieta[0] = Kokkos::atan(y/z);
	    xieta[1] = Kokkos::atan(-x/z);
	  } else {
	    xieta[0] = Kokkos::atan(-y/z);
	    xieta[1] = Kokkos::atan(-x/z);
	  }
	}
}

KOKKOS_INLINE_FUNCTION
void xieta_from_xyz(const double x, const double y, const double z, const int face, double* xieta) {
	if (face == 1) {
		xieta[0] = Kokkos::atan(y/x);
	    xieta[1] = Kokkos::atan(z/x);
	} else if (face == 2) {
		xieta[0] = Kokkos::atan(-x/y);
		xieta[1] = Kokkos::atan(z/y);
	} else if (face == 3) {
		xieta[0] = Kokkos::atan(y/x);
		xieta[1] = Kokkos::atan(-z/x);
	} else if (face == 4) {
		xieta[0] = Kokkos::atan(-x/y);
		xieta[1] = Kokkos::atan(-z/y);
	} else if (face == 5) {
		xieta[0] = Kokkos::atan(y/z);
		xieta[1] = Kokkos::atan(-x/z);
	} else if (face == 6) {
		xieta[0] = Kokkos::atan(-y/z);
		xieta[1] = Kokkos::atan(-x/z);
	} else {
		Kokkos::abort("Input face is not between 1 and 6, xieta_from_xyz");
	}
}

KOKKOS_INLINE_FUNCTION
void loncolatvec_from_xietavec(double& lon_comp, double& colat_comp, const double xi_deriv, const double eta_deriv, const int face, const double xi, const double eta) {
	lon_comp = 0;
	colat_comp = 0;
	double X = Kokkos::tan(xi);
	double Y = Kokkos::tan(eta);
	double X2 = X*X;
	double Y2 = Y*Y;
	double C = Kokkos::sqrt(1+X2);
	double D = Kokkos::sqrt(1+Y2);
	double grad_xi_comp = D * xi_deriv + X*Y/D * eta_deriv;
	double grad_eta_comp = X*Y/C*xi_deriv + C*eta_deriv;
	double delta = 1+X2+Y2;
	if ((face == 1) or (face == 2) or (face == 3) or (face == 4)) {
		colat_comp = X*Y/(C*D)*grad_xi_comp - grad_eta_comp;
		lon_comp = Kokkos::sqrt(delta)/(C*D)*grad_xi_comp;
	} else if (face == 5) {
		colat_comp = Kokkos::sqrt(X2+Y2)*(X/(D*(X2+Y2+1e-16))*grad_xi_comp + Y/(C*(X2+Y2+1e-16))*grad_eta_comp);
		lon_comp = Kokkos::sqrt(X2+Y2)*(-Kokkos::sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + Kokkos::sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
	} else if (face == 6) {
		colat_comp = -Kokkos::sqrt(X2+Y2)*(X/(D*(X2+Y2+1e-16))*grad_xi_comp + Y/(C*(X2+Y2+1e-16))*grad_eta_comp);
		lon_comp = -Kokkos::sqrt(X2+Y2)*(-Kokkos::sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + Kokkos::sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
	} else {
		Kokkos::abort("Input face is not between 1 and 6, lon colat vec from xi eta vec");
	}
}

KOKKOS_INLINE_FUNCTION
void xyzvec_from_xietavec(double& x_comp, double& y_comp, double& z_comp, const double xi_deriv, const double eta_deriv, const int face, const double xi, const double eta) {
	double lon_comp, colat_comp;
	double X = Kokkos::tan(xi);
	double Y = Kokkos::tan(eta);
	double X2=X*X;
	double Y2=Y*Y;
	double C=Kokkos::sqrt(1+X2);
	double D=Kokkos::sqrt(1+Y2);
	double grad_xi_comp = D*xi_deriv + X*Y/D*eta_deriv;
	double grad_eta_comp = X*Y/C*xi_deriv + C*eta_deriv;
	double delta=1+X2+Y2;
	double xyz[3], x, y, z;
	xyz_from_xieta(xi, eta, face, xyz);
	x = xyz[0];
	y = xyz[1];
	z = xyz[2];
	double sqc = Kokkos::sqrt(x*x+y*y);
	if ((face == 1) or (face == 2) or (face == 3) or (face == 4)) {
		colat_comp = X*Y/(C*D)*grad_xi_comp - grad_eta_comp;
		lon_comp = Kokkos::sqrt(delta)/(C*D)*grad_xi_comp;
		x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
		y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
		z_comp = -sqc*colat_comp;
	} else if (face == 5) {
		if ((Kokkos::abs(xi) < 1e-16) and (Kokkos::abs(eta) < 1e-16)) {
			// close to north pole
			z_comp = 0;
			x_comp = grad_xi_comp;
			y_comp = grad_eta_comp;
		} else {
			// away from north pole
			colat_comp = Kokkos::sqrt(X2+Y2)*(X/(D*(X2+Y2))*grad_xi_comp + Y/(C*(X2+Y2))*grad_eta_comp);
			lon_comp = Kokkos::sqrt(X2+Y2)*(-Kokkos::sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + Kokkos::sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
			x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
			y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
			z_comp = -sqc*colat_comp;
		}
	} else if (face == 6) {
		if ((Kokkos::abs(xi) < 1e-16) and (Kokkos::abs(eta) < 1e-16)) {
			// close to south pole
			z_comp = 0;
			x_comp = grad_xi_comp;
			y_comp = grad_eta_comp;
		} else {
			// away from south pole
			colat_comp = -Kokkos::sqrt(X2+Y2)*(X/(D*(X2+Y2))*grad_xi_comp + Y/(C*(X2+Y2))*grad_eta_comp);
			lon_comp = -Kokkos::sqrt(X2+Y2)*(-Kokkos::sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + Kokkos::sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
			x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
			y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
			z_comp = -sqc*colat_comp;
		}
	} else {
		Kokkos::abort("Input face is not between 1 and 6, xyz vec from xi eta vec");
	}
}

#endif