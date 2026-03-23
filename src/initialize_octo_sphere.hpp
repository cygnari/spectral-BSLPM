#ifndef H_INIT_OCTO_SPHERE_H
#define H_INIT_OCTO_SPHERE_H

#include<Kokkos_Core.hpp>

#include "run_config.hpp"

struct OctoSpherePanel {
	int id;
	int level;
	int parent_id;
	bool is_leaf;
	double min_lat; 
	double max_lat;
	double min_lon;
	double max_lon;
	double radius;
	double area;
	int point_count;
	int child1;
	int child2;
	int child3;
	int child4;
};

void initialize_octo_sphere(RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos,
											Kokkos::View<OctoSpherePanel*, Kokkos::HostSpace>& octo_sphere_tree, Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& leaf_panel_point_ids);

#endif