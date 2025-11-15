#include <cmath>
#include <stdexcept>

double cubed_sphere_jac(double xi, double eta) {
	return 1.0 / (pow(cos(xi) * cos(eta), 2) * pow(1 + pow(tan(xi), 2) + pow(tan(eta), 2), 1.5));
}

int face_from_xyz(const double x, const double y, const double z) {
	double ax = std::abs(x);
	double ay = std::abs(y);
	double az = std::abs(z);
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

void xyz_from_xieta(const double xi, const double eta, const int face, double* xyz) {
	double X = tan(xi);
	double Y = tan(eta);
	if (face == 0) {
	  xyz[0] = 1.0/sqrt(1+X*X+Y*Y);
	  xyz[1] = X*xyz[0];
	  xyz[2] = Y*xyz[0];
	} else if (face == 1) {
	  xyz[1] = 1.0/sqrt(1+X*X+Y*Y);
	  xyz[0] = -X*xyz[1];
	  xyz[2] = Y*xyz[1];
	} else if (face == 2) {
	  xyz[0] = -1.0/sqrt(1+X*X+Y*Y);
	  xyz[1] = X*xyz[0];
	  xyz[2] = -Y*xyz[0];
	} else if (face == 3) {
	  xyz[1] = -1.0/sqrt(1+X*X+Y*Y);
	  xyz[0] = -X*xyz[1];
	  xyz[2] = -Y*xyz[1];
	} else if (face == 4) {
	  xyz[2] = 1.0/sqrt(1+X*X+Y*Y);
	  xyz[0] = -Y*xyz[2];
	  xyz[1] = X*xyz[2];
	} else if (face == 5) {
	  xyz[2] = -1.0/sqrt(1+X*X+Y*Y);
	  xyz[0] = -Y*xyz[2];
	  xyz[1] = -X*xyz[2];
	} else {
		throw std::runtime_error("Input face is not between 1 and 6, xyz_from_xieta");
	}
}

void xieta_from_xyz(const double x, const double y, const double z, double* xieta) {
	double ax = std::abs(x);
	double ay = std::abs(y);
	double az = std::abs(z);

	if ((ax >= ay) and (ax >= az)) {
	  if (x >= 0) {
	    xieta[0] = atan(y/x);
	    xieta[1] = atan(z/x);
	  } else {
	    xieta[0] = atan(y/x);
        xieta[1] = atan(-z/x);
	  }
	} else if ((ay >= ax) and (ay >= az)) {
	  if (y >= 0) {
	    xieta[0] = atan(-x/y);
        xieta[1] = atan(z/y);
	  } else {
	    xieta[0] = atan(-x/y);
	    xieta[1] = atan(-z/y);
	  }
	} else {
	  if (z >= 0) {
	    xieta[0] = atan(y/z);
	    xieta[1] = atan(-x/z);
	  } else {
	    xieta[0] = atan(-y/z);
	    xieta[1] = atan(-x/z);
	  }
	}
}

void xieta_from_xyz(const double x, const double y, const double z, const int face, double* xieta) {
	if (face == 0) {
		xieta[0] = atan(y/x);
	    xieta[1] = atan(z/x);
	} else if (face == 1) {
	    xieta[0] = atan(-x/y);
	    xieta[1] = atan(z/y);
	} else if (face == 2) {
	    xieta[0] = atan(y/x);
	    xieta[1] = atan(-z/x);
	} else if (face == 3) {
		xieta[0] = atan(-x/y);
		xieta[1] = atan(-z/y);
	} else if (face == 4) {
		xieta[0] = atan(y/z);
		xieta[1] = atan(-x/z);
	} else if (face == 5) {
		xieta[0] = atan(-y/z);
		xieta[1] = atan(-x/z);
	} else {
		throw std::runtime_error("Input face is not between 1 and 6, xieta_from_xyz");
	}
}

void loncolatvec_from_xietavec(double& lon_comp, double& colat_comp, const double xi_deriv, const double eta_deriv, const int face, const double xi, const double eta) {
	lon_comp = 0;
	colat_comp = 0;
	double X = tan(xi);
	double Y = tan(eta);
	double X2 = X*X;
	double Y2 = Y*Y;
	double C = sqrt(1+X2);
	double D = sqrt(1+Y2);
	double grad_xi_comp = D * xi_deriv + X*Y/D * eta_deriv;
	double grad_eta_comp = X*Y/C*xi_deriv + C*eta_deriv;
	double delta = 1+X2+Y2;
	if ((face == 1) or (face == 2) or (face == 3) or (face == 4)) {
		colat_comp = X*Y/(C*D)*grad_xi_comp - grad_eta_comp;
		lon_comp = sqrt(delta)/(C*D)*grad_xi_comp;
	} else if (face == 5) {
		colat_comp = sqrt(X2+Y2)*(X/(D*(X2+Y2+1e-16))*grad_xi_comp + Y/(C*(X2+Y2+1e-16))*grad_eta_comp);
		lon_comp = sqrt(X2+Y2)*(-sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
	} else if (face == 6) {
		colat_comp = -sqrt(X2+Y2)*(X/(D*(X2+Y2+1e-16))*grad_xi_comp + Y/(C*(X2+Y2+1e-16))*grad_eta_comp);
		lon_comp = -sqrt(X2+Y2)*(-sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
	} else {
		throw std::runtime_error("Input face is not between 1 and 6, lon colat vec from xi eta vec");
	}
}

void xyzvec_from_xietavec(double& x_comp, double& y_comp, double& z_comp, const double xi_deriv, const double eta_deriv, const int face, const double xi, const double eta) {
	double lon_comp, colat_comp;
	double X = tan(xi);
	double Y = tan(eta);
	double X2=X*X;
	double Y2=Y*Y;
	double C=sqrt(1+X2);
	double D=sqrt(1+Y2);
	double grad_xi_comp = D*xi_deriv + X*Y/D*eta_deriv;
	double grad_eta_comp = X*Y/C*xi_deriv + C*eta_deriv;
	double delta=1+X2+Y2;
	double xyz[3], x, y, z;
	xyz_from_xieta(xi, eta, face, xyz);
	x = xyz[0];
	y = xyz[1];
	z = xyz[2];
	double sqc = sqrt(x*x+y*y);
	if ((face == 1) or (face == 2) or (face == 3) or (face == 4)) {
		colat_comp = X*Y/(C*D)*grad_xi_comp - grad_eta_comp;
		lon_comp = sqrt(delta)/(C*D)*grad_xi_comp;
		x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
		y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
		z_comp = -sqc*colat_comp;
	} else if (face == 5) {
		if ((std::abs(xi) < 1e-16) and (std::abs(eta) < 1e-16)) {
			// close to north pole
			z_comp = 0;
			x_comp = grad_xi_comp;
			y_comp = grad_eta_comp;
		} else {
			// away from north pole
			colat_comp = sqrt(X2+Y2)*(X/(D*(X2+Y2))*grad_xi_comp + Y/(C*(X2+Y2))*grad_eta_comp);
			lon_comp = sqrt(X2+Y2)*(-sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
			x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
			y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
			z_comp = -sqc*colat_comp;
		}
	} else if (face == 6) {
		if ((std::abs(xi) < 1e-16) and (std::abs(eta) < 1e-16)) {
			// close to south pole
			z_comp = 0;
			x_comp = grad_xi_comp;
			y_comp = grad_eta_comp;
		} else {
			// away from south pole
			colat_comp = -sqrt(X2+Y2)*(X/(D*(X2+Y2))*grad_xi_comp + Y/(C*(X2+Y2))*grad_eta_comp);
			lon_comp = -sqrt(X2+Y2)*(-sqrt(delta)*Y/(D*(X2+Y2))*grad_xi_comp + sqrt(delta)*X/(C*(X2+Y2))*grad_eta_comp);
			x_comp = x*z/sqc * colat_comp - y/sqc * lon_comp;
			y_comp = y*z/sqc * colat_comp + x/sqc * lon_comp;
			z_comp = -sqc*colat_comp;
		}
	} else {
		throw std::runtime_error("Input face is not between 1 and 6, xyz vec from xi eta vec");
	}
}
