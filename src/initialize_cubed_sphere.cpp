#include<Kokkos_Core.hpp>

#include "cubed_sphere_transforms.hpp"
#include "cubed_sphere_transforms_impl.hpp"
#include "general_utils.hpp"
#include "general_utils_impl.hpp"
#include "run_config.hpp"

struct compute_point {
	RunConfig run_config;
	Kokkos::View<double*, Kokkos::HostSpace> xcos;
	Kokkos::View<double*, Kokkos::HostSpace> ycos;
	Kokkos::View<double*, Kokkos::HostSpace> zcos;
	Kokkos::View<double*, Kokkos::HostSpace> area;
	double small_offset;
	double dist_between;
	int points_per_side;

	compute_point(const RunConfig& run_config_, Kokkos::View<double*, Kokkos::HostSpace>& xcos_, Kokkos::View<double*, Kokkos::HostSpace>& ycos_, Kokkos::View<double*, Kokkos::HostSpace>& zcos_, Kokkos::View<double*, Kokkos::HostSpace>& area_, double small_offset_, double dist_between_, int points_per_side_) :
					run_config(run_config_), xcos(xcos_), ycos(ycos_), zcos(zcos_), area(area_), dist_between(dist_between_), small_offset(small_offset_), points_per_side(points_per_side_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j, const int k) const {
		double xi = -Kokkos::numbers::pi / 4.0 + small_offset + dist_between * j;
		double eta = -Kokkos::numbers::pi / 4.0 + small_offset + dist_between * k;

		double xyz[3], xyz1[3], xyz2[3], xyz3[3], xyz4[3], area1, area2;
		xyz_from_xieta(xi, eta, i+1, xyz);
		xyz_from_xieta(xi-small_offset, eta-small_offset, i+1, xyz1);
		xyz_from_xieta(xi+small_offset, eta-small_offset, i+1, xyz2);
		xyz_from_xieta(xi+small_offset, eta+small_offset, i+1, xyz3);
		xyz_from_xieta(xi-small_offset, eta+small_offset, i+1, xyz4);
		int point_loc;
		point_loc = i * points_per_side * points_per_side + j * points_per_side + k;
		xcos(point_loc) = xyz[0];
		ycos(point_loc) = xyz[1];
		zcos(point_loc) = xyz[2];
		area1 = sphere_tri_area(xyz1, xyz2, xyz3);
		area2 = sphere_tri_area(xyz1, xyz3, xyz4);
		area(point_loc) = area1 + area2;
	}
};

void cubed_sphere_midpoints(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& xcos, Kokkos::View<double*, Kokkos::HostSpace>& ycos, Kokkos::View<double*, Kokkos::HostSpace>& zcos, Kokkos::View<double*, Kokkos::HostSpace>& area) {
	int points_per_side = Kokkos::pow(2, run_config.levels);

	double dist_between = M_PI / (2.0*points_per_side);
	double small_offset = dist_between/2.0;

	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultHostExecutionSpace(), {0,0,0},{6,points_per_side,points_per_side}), compute_point(run_config, xcos, ycos, zcos, area, small_offset, dist_between, points_per_side));
	Kokkos::fence();
}
