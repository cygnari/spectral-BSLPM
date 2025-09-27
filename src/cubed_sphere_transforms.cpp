#include <cmath>

double cubed_sphere_jac(double xi, double eta) {
	return 1.0 / (pow(cos(xi) * cos(eta), 2) * pow(1 + pow(tan(xi), 2) + pow(tan(eta), 2), 1.5));
}