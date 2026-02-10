#ifndef H_CUBE_SPHERE_TRANSFORMS_IMPL_H
#define H_CUBE_SPHERE_TRANSFORMS_IMPL_H

#include <Kokkos_Core.hpp>
#include <KokkosBatched_Gesv.hpp>
#include <iostream>

KOKKOS_INLINE_FUNCTION
int face_from_xyz(const double x, const double y, const double z) {
	double ax = Kokkos::abs(x);
	double ay = Kokkos::abs(y);
	double az = Kokkos::abs(z);
	if ((ax >= ay) and (ax >= az)) {
	  if (x >= 0) {
	    return 0;
	  } else {
	    return 2;
	  }
	} else if ((ay >= ax) and (ay >= az)) {
	  if (y >= 0) {
	    return 1;
	  } else {
	    return 3;
	  }
	} else {
	  if (z >= 0) {
	    return 4;
	  } else {
	    return 5;
	  }
	}
}

KOKKOS_INLINE_FUNCTION
void xyz_from_xieta(const double xi, const double eta, const int face, double* xyz) {
	double X = Kokkos::tan(xi);
	double Y = Kokkos::tan(eta);
	if (face == 0) {
	  xyz[0] = 1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[1] = X*xyz[0];
	  xyz[2] = Y*xyz[0];
	} else if (face == 1) {
	  xyz[1] = 1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[0] = -X*xyz[1];
	  xyz[2] = Y*xyz[1];
	} else if (face == 2) {
	  xyz[0] = -1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[1] = X*xyz[0];
	  xyz[2] = -Y*xyz[0];
	} else if (face == 3) {
	  xyz[1] = -1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[0] = -X*xyz[1];
	  xyz[2] = -Y*xyz[1];
	} else if (face == 4) {
	  xyz[2] = 1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[0] = -Y*xyz[2];
	  xyz[1] = X*xyz[2];
	} else if (face == 5) {
	  xyz[2] = -1.0/Kokkos::sqrt(1+X*X+Y*Y);
	  xyz[0] = -Y*xyz[2];
	  xyz[1] = -X*xyz[2];
	} else {
	  Kokkos::abort("Input face is not between 0 and 5, xyz_from_xieta");
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
	if (face == 0) {
		xieta[0] = Kokkos::atan(y/x);
	    xieta[1] = Kokkos::atan(z/x);
	} else if (face == 1) {
		xieta[0] = Kokkos::atan(-x/y);
		xieta[1] = Kokkos::atan(z/y);
	} else if (face == 2) {
		xieta[0] = Kokkos::atan(y/x);
		xieta[1] = Kokkos::atan(-z/x);
	} else if (face == 3) {
		xieta[0] = Kokkos::atan(-x/y);
		xieta[1] = Kokkos::atan(-z/y);
	} else if (face == 4) {
		xieta[0] = Kokkos::atan(y/z);
		xieta[1] = Kokkos::atan(-x/z);
	} else if (face == 5) {
		xieta[0] = Kokkos::atan(-y/z);
		xieta[1] = Kokkos::atan(-x/z);
	} else {
		Kokkos::abort("Input face is not between 0 and 5, xieta_from_xyz");
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
	if ((face == 0) or (face == 1) or (face == 2) or (face == 3)) {
		colat_comp = X*Y/(C*D)*grad_xi_comp - grad_eta_comp;
		lon_comp = Kokkos::sqrt(delta)/(C*D)*grad_xi_comp;
	} else if (face == 4) {
		colat_comp = Kokkos::sqrt(X2+Y2)*(X/(D*(X2+Y2+1e-16))*grad_xi_comp + Y/(C*(X2+Y2+1e-16))*grad_eta_comp);
		lon_comp = Kokkos::sqrt(X2+Y2)*(-Kokkos::sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + Kokkos::sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
	} else if (face == 5) {
		colat_comp = -Kokkos::sqrt(X2+Y2)*(X/(D*(X2+Y2+1e-16))*grad_xi_comp + Y/(C*(X2+Y2+1e-16))*grad_eta_comp);
		lon_comp = -Kokkos::sqrt(X2+Y2)*(-Kokkos::sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + Kokkos::sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
	} else {
		Kokkos::abort("Input face is not between 0 and 5, lon colat vec from xi eta vec");
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
	double isqd = 1.0/Kokkos::sqrt(delta);
	double xyz[3], x, y, z;
	double mat_buf[4], work_buf[12], rhs[2], sol[2];
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Unmanaged>> own_mat(mat_buf, 2, 2);
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Unmanaged>> own_temp(work_buf, 2, 6);
	Kokkos::View<double[2], Kokkos::MemoryTraits<Kokkos::Unmanaged>> own_rhs(rhs, 2);
	Kokkos::View<double[2], Kokkos::MemoryTraits<Kokkos::Unmanaged>> own_sol(sol, 2); 
	xyz_from_xieta(xi, eta, face, xyz);
	x = xyz[0];
	y = xyz[1];
	z = xyz[2];
	double sqc = Kokkos::sqrt(x*x+y*y);
	double isqc2 = 1.0/Kokkos::sqrt(X2+Y2);
	own_rhs(0) = grad_xi_comp;
	own_rhs(1) = grad_eta_comp;
	if ((face == 0) or (face == 1) or (face == 2) or (face == 3)) {
		own_mat(0,0) = 0;
		own_mat(0,1) = C*D*isqd;
		own_mat(1,0) = -1.0;
		own_mat(1,1) = X*Y*isqd;
		
		KokkosBatched::SerialGesv<KokkosBatched::Gesv::StaticPivoting>::invoke(own_mat, own_sol, own_rhs, own_temp);
		colat_comp = own_sol(0);
		lon_comp = own_sol(1);
		// colat_comp = X*Y/(C*D)*grad_xi_comp - grad_eta_comp;
		// lon_comp = Kokkos::sqrt(delta)/(C*D)*grad_xi_comp;
		x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
		y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
		z_comp = -sqc*colat_comp;
	} else if (face == 4) {
		if ((Kokkos::abs(xi) < 1e-16) and (Kokkos::abs(eta) < 1e-16)) {
			// close to north pole
			z_comp = 0;
			x_comp = grad_xi_comp;
			y_comp = grad_eta_comp;
		} else {
			// away from north pole
			own_mat(0,0) = isqc2*D*X;
			own_mat(0,1) = isqc2*-D*Y*isqd;
			own_mat(1,0) = isqc2*C*Y;
			own_mat(1,1) = isqc2*C*X*isqd;
			KokkosBatched::SerialGesv<KokkosBatched::Gesv::StaticPivoting>::invoke(own_mat, own_sol, own_rhs, own_temp);
			colat_comp = own_sol(0);
			lon_comp = own_sol(1);

			// colat_comp = Kokkos::sqrt(X2+Y2)*(X/(D*(X2+Y2))*grad_xi_comp + Y/(C*(X2+Y2))*grad_eta_comp);
			// lon_comp = Kokkos::sqrt(X2+Y2)*(-Kokkos::sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + Kokkos::sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
			x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
			y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
			z_comp = -sqc*colat_comp;
		}
	} else if (face == 5) {
		if ((Kokkos::abs(xi) < 1e-16) and (Kokkos::abs(eta) < 1e-16)) {
			// close to south pole
			z_comp = 0;
			x_comp = grad_xi_comp;
			y_comp = grad_eta_comp;
		} else {
			// away from south pole
			own_mat(0,0) = -isqc2*D*X;
			own_mat(0,1) = isqc2*D*Y*isqd;
			own_mat(1,0) = -isqc2*C*Y;
			own_mat(1,1) = -isqc2*C*X*isqd;
			KokkosBatched::SerialGesv<KokkosBatched::Gesv::StaticPivoting>::invoke(own_mat, own_sol, own_rhs, own_temp);
			colat_comp = own_sol(0);
			lon_comp = own_sol(1);
			// colat_comp = -Kokkos::sqrt(X2+Y2)*(X/(D*(X2+Y2))*grad_xi_comp + Y/(C*(X2+Y2))*grad_eta_comp);
			// lon_comp = -Kokkos::sqrt(X2+Y2)*(-Kokkos::sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + Kokkos::sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
			x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
			y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
			z_comp = -sqc*colat_comp;
		}
	} else {
		Kokkos::abort("Input face is not between 0 and 5, xyz vec from xi eta vec");
	}
}

KOKKOS_INLINE_FUNCTION
void xietavec_from_xyzvec(double& xi_comp, double& eta_comp, double x_comp, double y_comp, double z_comp, double x, double y, double z) {
	double xieta[2], colat_comp, lon_comp;
	int face;
	xieta_from_xyz(x, y, z, xieta);
	face = face_from_xyz(x, y, z);
	double X, Y, X2, Y2, C, D, delta;
	if (Kokkos::abs(z) < 1-1e-16) {
		// away from poles
		lon_comp = (-y*x_comp + x*y_comp) / Kokkos::sqrt(x*x+y*y);
		colat_comp = ((x*x_comp + y*y_comp)*z -(x*x+y*y)*z_comp) / Kokkos::sqrt(x*x+y*y);
		X = Kokkos::tan(xieta[0]);
		Y = Kokkos::tan(xieta[1]);
		X2 = X*X;
		Y2 = Y*Y;
		C = Kokkos::sqrt(1+X2);
		D = Kokkos::sqrt(1+Y2);
		delta = 1+X2+Y2;
		if ((face == 0) or (face == 1) or (face == 2) or (face == 3)) {
			xi_comp = C*D/Kokkos::sqrt(delta)*lon_comp;
			eta_comp = -1.0*colat_comp + X*Y/Kokkos::sqrt(delta)*lon_comp;
		} else if (face == 4) {
			xi_comp = 1.0/Kokkos::sqrt(X2+Y2)*(D*X*colat_comp-D*Y/Kokkos::sqrt(delta)*lon_comp);
			eta_comp = 1.0/Kokkos::sqrt(X2+Y2)*(C*Y*colat_comp+C*X/Kokkos::sqrt(delta)*lon_comp);
		} else if (face == 5) {
			xi_comp = -1.0/Kokkos::sqrt(X2+Y2)*(D*X*colat_comp-D*Y/Kokkos::sqrt(delta)*lon_comp);
			eta_comp = -1.0/Kokkos::sqrt(X2+Y2)*(C*Y*colat_comp+C*X/Kokkos::sqrt(delta)*lon_comp);
		} else {
			Kokkos::abort("Input face is not between 0 and 5, xi eta vec from xyz vec");
		}
	} else {
		// close to poles
		xi_comp = x_comp;
		eta_comp = y_comp;
	}
}

KOKKOS_INLINE_FUNCTION
void xietavec_from_xyzvec_2(double& xi_comp, double& eta_comp, double x_comp, double y_comp, double z_comp, double x, double y, double z) {
	double xieta[2], colat_comp, lon_comp;
	int face;
	xieta_from_xyz(x, y, z, xieta);
	std::cout << xieta[0] << " " << xieta[1] << std::endl;
	face = face_from_xyz(x, y, z);
	std::cout << face << std::endl;
	double X, Y, X2, Y2, C, D, delta;
	if (Kokkos::abs(z) < 1-1e-16) {
		// away from poles
		lon_comp = (-y*x_comp + x*y_comp) / Kokkos::sqrt(x*x+y*y);
		colat_comp = ((x*x_comp + y*y_comp)*z -(x*x+y*y)*z_comp) / Kokkos::sqrt(x*x+y*y);
		std::cout << lon_comp << " " << colat_comp << std::endl;
		X = Kokkos::tan(xieta[0]);
		Y = Kokkos::tan(xieta[1]);
		X2 = X*X;
		Y2 = Y*Y;
		C = Kokkos::sqrt(1+X2);
		D = Kokkos::sqrt(1+Y2);
		delta = 1+X2+Y2;
		std::cout << X << " " << Y << " " << C << " " << D << " " << delta << std::endl;
		if ((face == 0) or (face == 1) or (face == 2) or (face == 3)) {
			xi_comp = C*D/Kokkos::sqrt(delta)*lon_comp;
			eta_comp = -1.0*colat_comp + X*Y/Kokkos::sqrt(delta)*lon_comp;
		} else if (face == 4) {
			xi_comp = 1.0/Kokkos::sqrt(X2+Y2)*(D*X*colat_comp-D*Y/Kokkos::sqrt(delta)*lon_comp);
			eta_comp = 1.0/Kokkos::sqrt(X2+Y2)*(C*Y*colat_comp+C*X/Kokkos::sqrt(delta)*lon_comp);
		} else if (face == 5) {
			xi_comp = -1.0/Kokkos::sqrt(X2+Y2)*(D*X*colat_comp-D*Y/Kokkos::sqrt(delta)*lon_comp);
			eta_comp = -1.0/Kokkos::sqrt(X2+Y2)*(C*Y*colat_comp+C*X/Kokkos::sqrt(delta)*lon_comp);
		} else {
			Kokkos::abort("Input face is not between 0 and 5, xi eta vec from xyz vec");
		}
	} else {
		// close to poles
		xi_comp = x_comp;
		eta_comp = y_comp;
	}
}

#endif