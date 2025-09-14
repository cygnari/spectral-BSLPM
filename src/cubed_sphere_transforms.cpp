// #include <cmath>
// #include <stdexcept>

// int face_from_xyz(const double x, const double y, const double z) {
// 	double ax = std::abs(x);
// 	double ay = std::abs(y);
// 	double az = std::abs(z);
// 	if ((ax >= ay) and (ax >= az)) {
// 	  if (x >= 0) {
// 	    return 1;
// 	  } else {
// 	    return 3;
// 	  }
// 	} else if ((ay >= ax) and (ay >= az)) {
// 	  if (y >= 0) {
// 	    return 2;
// 	  } else {
// 	    return 4;
// 	  }
// 	} else {
// 	  if (z >= 0) {
// 	    return 5;
// 	  } else {
// 	    return 6;
// 	  }
// 	}
// }

// double* xyz_from_xieta_1(const double xi, const double eta, double* xyz) {
//   double X = tan(xi);
//   double Y = tan(eta);
//   // double xyz[3];
//   xyz[0] = 1/sqrt(1+X*X+Y*Y);
//   xyz[1] = X*xyz[0];
//   xyz[2] = Y*xyz[0];
//   return xyz;
// }

// double* xyz_from_xieta_2(const double xi, const double eta, double* xyz) {
//   double X = tan(xi);
//   double Y = tan(eta);
//   // double xyz[3];
//   xyz[1] = 1/sqrt(1+X*X+Y*Y);
//   xyz[0] = -X*xyz[1];
//   xyz[2] = Y*xyz[1];
//   return xyz;
// }

// double* xyz_from_xieta_3(const double xi, const double eta, double* xyz) {
//   double X = tan(xi);
//   double Y = tan(eta);
//   // double xyz[3];
//   xyz[0] = -1/sqrt(1+X*X+Y*Y);
//   xyz[1] = X*xyz[0];
//   xyz[2] = -Y*xyz[0];
//   return xyz;
// }

// double* xyz_from_xieta_4(const double xi, const double eta, double* xyz) {
//   double X = tan(xi);
//   double Y = tan(eta);
//   // double xyz[3];
//   xyz[1] = -1/sqrt(1+X*X+Y*Y);
//   xyz[0] = -X*xyz[1];
//   xyz[2] = -Y*xyz[1];
//   return xyz;
// }

// double* xyz_from_xieta_5(const double xi, const double eta, double* xyz) {
//   double X = tan(xi);
//   double Y = tan(eta);
//   // double xyz[3];
//   xyz[2] = 1/sqrt(1+X*X+Y*Y);
//   xyz[0] = -Y*xyz[2];
//   xyz[1] = X*xyz[2];
//   return xyz;
// }

// double* xyz_from_xieta_6(const double xi, const double eta, double* xyz) {
//   double X = tan(xi);
//   double Y = tan(eta);
//   // double xyz[3];
//   xyz[2] = -1/sqrt(1+X*X+Y*Y);
//   xyz[0] = -Y*xyz[2];
//   xyz[1] = -X*xyz[2];
//   return xyz;
// }

// double* xyz_from_xieta(const double xi, const double eta, const int face, double* xyz) {
// 	if (face == 1) {
// 	  return xyz_from_xieta_1(xi, eta, xyz);
// 	} else if (face == 2) {
// 	  return xyz_from_xieta_2(xi, eta, xyz);
// 	} else if (face == 3) {
// 	  return  xyz_from_xieta_3(xi, eta, xyz);
// 	} else if (face == 4) {
// 	  return xyz_from_xieta_4(xi, eta, xyz);
// 	} else if (face == 5) {
// 	  return xyz_from_xieta_5(xi, eta, xyz);
// 	} else if (face == 6) {
// 	  return xyz_from_xieta_6(xi, eta, xyz);
// 	} else {
// 	  throw std::runtime_error("face is not 1 to 6, xyz_from_xieta");
// 	}
// }

// double* xieta_from_xyz_1(const double x, const double y, const double z, double* xieta) {
//   // double xieta[2];
//   xieta[0] = atan(y/x);
//   xieta[1] = atan(z/x);
//   return xieta;
// }

// double* xieta_from_xyz_2(const double x, const double y, const double z, double* xieta) {
//   // double xieta[2];
//   xieta[0] = atan(-x/y);
//   xieta[1] = atan(z/y);
//   return xieta;
// }

// double* xieta_from_xyz_3(const double x, const double y, const double z, double* xieta) {
//   // double xieta[2];
//   xieta[0] = atan(y/x);
//   xieta[1] = atan(-z/x);
//   return xieta;
// }

// double* xieta_from_xyz_4(const double x, const double y, const double z, double* xieta) {
//   // double xieta[2];
//   xieta[0] = atan(-x/y);
//   xieta[1] = atan(-z/y);
//   return xieta;
// }

// double* xieta_from_xyz_5(const double x, const double y, const double z, double* xieta) {
//   // double xieta[2];
//   xieta[0] = atan(y/z);
//   xieta[1] = atan(-x/z);
//   return xieta;
// }

// double* xieta_from_xyz_6(const double x, const double y, const double z, double* xieta) {
//   // double xieta[2];
//   xieta[0] = atan(-y/z);
//   xieta[1] = atan(-x/z);
//   return xieta;
// }

// double* xieta_from_xyz(const double x, const double y, const double z, double* xieta) {
// 	double ax = std::abs(x);
// 	double ay = std::abs(y);
// 	double az = std::abs(z);

// 	if ((ax >= ay) and (ax >= az)) {
// 	  if (x >= 0) {
// 	    return xieta_from_xyz_1(x, y, z, xieta);
// 	  } else {
// 	    return xieta_from_xyz_3(x, y, z, xieta);
// 	  }
// 	} else if ((ay >= ax) and (ay >= az)) {
// 	  if (y >= 0) {
// 	    return xieta_from_xyz_2(x, y, z, xieta);
// 	  } else {
// 	    return xieta_from_xyz_4(x, y, z, xieta);
// 	  }
// 	} else {
// 	  if (z >= 0) {
// 	    return xieta_from_xyz_5(x, y, z, xieta);
// 	  } else {
// 	    return xieta_from_xyz_6(x, y, z, xieta);
// 	  }
// 	}
// }

// double* xieta_from_xyz(const double x, const double y, const double z, const int face, double* xieta) {
// 	if (face == 1) {
// 	  return xieta_from_xyz_1(x, y, z, xieta);
// 	} else if (face == 2) {
// 	  return xieta_from_xyz_2(x, y, z, xieta);
// 	} else if (face == 3) {
// 	  return xieta_from_xyz_3(x, y, z, xieta);
// 	} else if (face == 4) {
// 	  return xieta_from_xyz_4(x, y, z, xieta);
// 	} else if (face == 5) {
// 	  return xieta_from_xyz_5(x, y, z, xieta);
// 	} else if (face == 6) {
// 	  return xieta_from_xyz_6(x, y, z, xieta);
// 	} else {
// 	  throw std::runtime_error("Face not between 1 and 6, xieta_from_xyz");
// 	}
// }