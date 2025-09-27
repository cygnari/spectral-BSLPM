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

void solution_2d_to_1d(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& area_1d, Kokkos::View<double*, Kokkos::HostSpace>& pots_1d, 
						Kokkos::View<double*, Kokkos::HostSpace>& soln_1d, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& pots, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& soln, 
						Kokkos::View<int*, Kokkos::HostSpace>& one_d_no_of_points, Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d);

#endif
