#ifndef H_INIT_CUBE_SPHERE_H
#define H_INIT_CUBE_SPHERE_H

#include<Kokkos_Core.hpp>

#include "run_config.hpp"

struct CubedSpherePanel {
	int id;
	int id_top_edge; // eta=1
	int id_left_edge; // xi=-1
	int id_bot_edge; // eta=-1
	int id_right_edge; // xi=1
	int face;
	int level;
	int parent_id;
	int child1;
	int child2;
	int child3;
	int child4;
	bool is_leaf;
	double min_xi; // -1 to 1, multiplied by pi/4 when converting to sphere
	double max_xi;
	double min_eta;
	double max_eta;
	double radius;
	double area;
};

void cubed_sphere_midpoints(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& xcos, 
							Kokkos::View<double*, Kokkos::HostSpace>& ycos, Kokkos::View<double*, 
							Kokkos::HostSpace>& zcos, Kokkos::View<double*, Kokkos::HostSpace>& area);

void cubed_sphere_panels_init(const RunConfig& run_config, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels);

void cube_sphere_spec_points(const RunConfig& run_config, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_vals, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area);

void cubed_sphere_2d_to_1d(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& xcos_1d, Kokkos::View<double*, Kokkos::HostSpace>& ycos_1d,
							Kokkos::View<double*, Kokkos::HostSpace>& zcos_1d, Kokkos::View<int*, Kokkos::HostSpace>& one_d_no_of_points, 
							Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos);

void unify_boundary_vals(const RunConfig& run_config, Kokkos::View<int*>& one_d_no_of_points, Kokkos::View<int**, Kokkos::LayoutRight>& two_d_to_1d, Kokkos::View<double**, Kokkos::LayoutRight>& vals);

void solution_2d_to_1d(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& area_1d, Kokkos::View<double*, Kokkos::HostSpace>& pots_1d, 
						Kokkos::View<double*, Kokkos::HostSpace>& soln_1d, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& pots, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& soln, 
						Kokkos::View<int*, Kokkos::HostSpace>& one_d_no_of_points, Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d);

void vec_2d_to_1d(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& vec_1d, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vec, 
						Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d, bool add);

template <class LayoutType> struct v_2d_to_1d{
	Kokkos::View<double*, Kokkos::HostSpace> vec_1d;
	Kokkos::View<double**, LayoutType, Kokkos::HostSpace> vec;
	Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace> two_d_to_1d;
	bool add;

	v_2d_to_1d(Kokkos::View<double*, Kokkos::HostSpace>& vec_1d_, Kokkos::View<double**, LayoutType, Kokkos::HostSpace>& vec_, 
					Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d_, bool add_) : vec_1d(vec_1d_), vec(vec_), two_d_to_1d(two_d_to_1d_), add(add_) {}

	void operator()(const int i, const int j) const {
		int loc = two_d_to_1d(i,j);
		if (add) {
			Kokkos::atomic_add(&vec_1d(loc), vec(i,j));
		} else {
			if (vec_1d(loc) == 0) {
				Kokkos::atomic_add(&vec_1d(loc), vec(i,j));
			}
		}
	}
};

template <class LayoutType> void vec_2d_to_1d(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& vec_1d, Kokkos::View<double**, LayoutType, Kokkos::HostSpace>& vec, 
						Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d, bool add) {
	for (int i = 0; i < vec_1d.extent_int(0); i++) {
		vec_1d(i) = 0;
	}
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultHostExecutionSpace(), {0, 0}, {vec.extent_int(0), vec.extent_int(1)}), v_2d_to_1d<LayoutType>(vec_1d, vec, two_d_to_1d, add));
}

#endif
