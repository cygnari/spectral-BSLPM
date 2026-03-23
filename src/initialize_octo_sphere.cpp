#include<Kokkos_Core.hpp>
#include <iostream>

#include "general_utils.hpp"
#include "initialize_octo_sphere.hpp"
#include "run_config.hpp"
#include "general_utils_impl.hpp"

void initialize_octo_sphere(RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos,
											Kokkos::View<OctoSpherePanel*, Kokkos::HostSpace>& octo_sphere_tree, Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& leaf_panel_point_ids) {
	std::vector<OctoSpherePanel> octo_panels(8);
	std::vector<std::vector<int>> points_inside(8);

	double lat, lon, lat_d, lon_d, min_lat, max_lat, min_lon, max_lon, mid_lat, mid_lon;
	int index, index1, points_to_assign, index_i, index_j, which_panel, point_count, start;

	for (int i = 0; i < 4; i++) {
		octo_panels[i].id = i;
		octo_panels[i].level = 0;
		octo_panels[i].parent_id = -1;
		octo_panels[i].min_lon = i*90.0;
		octo_panels[i].max_lon = (i+1)*90.0;
		octo_panels[i].min_lat = -90.0;
		octo_panels[i].max_lat = 0;
		octo_panels[i].is_leaf = true;
		octo_panels[i].point_count = 0;
	}
	for (int i = 4; i < 8; i++) {
		octo_panels[i].id = i;
		octo_panels[i].level = 0;
		octo_panels[i].parent_id = -1;
		octo_panels[i].min_lon = (i-4)*90.0;
		octo_panels[i].max_lon = (i-3)*90.0;
		octo_panels[i].min_lat = 0;
		octo_panels[i].max_lat = 90.0;
		octo_panels[i].is_leaf = true;
		octo_panels[i].point_count = 0;
	}

	for (int i = 0; i < run_config.lat_count; i++) {
		for (int j = 0; j < run_config.lon_count; j++) {
			index = 0;
			index1 = i*run_config.lon_count + j;
			xyz_to_latlon(lat, lon, xcos(i,j), ycos(i,j), zcos(i,j));

			if (lat > 0) {
				index += 4;
			} 
			lon = std::fmod(lon + 2.0*M_PI, 2.0*M_PI);
			index += std::floor(lon / (0.5*M_PI));
			octo_panels[index].point_count += 1;
			points_inside[index].push_back(index1);
		}
	}

	for (int i = 0; i < octo_panels.size(); i++) {
		if (octo_panels[i].point_count > run_config.leaf_size) {
			octo_panels[i].is_leaf = false;
			OctoSpherePanel sub_panel1, sub_panel2, sub_panel3, sub_panel4;
			min_lat = octo_panels[i].min_lat;
			max_lat = octo_panels[i].max_lat;
			min_lon = octo_panels[i].min_lon;
			max_lon = octo_panels[i].max_lon;
			mid_lat = 0.5*(min_lat + max_lat);
			mid_lon = 0.5*(min_lon + max_lon);
			sub_panel1.min_lon = mid_lon, sub_panel1.max_lon = max_lon, sub_panel1.min_lat = mid_lat, sub_panel1.max_lat = max_lat;
			sub_panel2.min_lon = min_lon, sub_panel2.max_lon = mid_lon, sub_panel2.min_lat = mid_lat, sub_panel2.max_lat = max_lat;
			sub_panel3.min_lon = min_lon, sub_panel3.max_lon = mid_lon, sub_panel3.min_lat = min_lat, sub_panel3.max_lat = mid_lat;
			sub_panel4.min_lon = mid_lon, sub_panel4.max_lon = max_lon, sub_panel4.min_lat = min_lat, sub_panel4.max_lat = mid_lat;
			start = octo_panels.size();
			octo_panels.push_back(sub_panel1);
			octo_panels.push_back(sub_panel2);
			octo_panels.push_back(sub_panel3);
			octo_panels.push_back(sub_panel4);
			for (int j = 0; j < 4; j++) {
				octo_panels[start+j].level = octo_panels[i].level + 1;
				octo_panels[start+j].id = start+j;
				octo_panels[start+j].parent_id = octo_panels[i].id;
				octo_panels[start+j].point_count = 0;
				octo_panels[start+j].is_leaf = true;
			}
			octo_panels[i].child1 = start;
			octo_panels[i].child2 = start+1;
			octo_panels[i].child3 = start+2;
			octo_panels[i].child4 = start+3;

			points_to_assign = octo_panels[i].point_count;
			points_inside.resize(start+4);
			for (int j = 0; j < points_to_assign; j++) {
				index = points_inside[i][j];
				index_j = index % run_config.lon_count;
				index_i = index / run_config.lon_count;
				xyz_to_latlon(lat, lon, xcos(index_i,index_j), ycos(index_i,index_j), zcos(index_i,index_j));
				lon = std::fmod(lon + 2.0*M_PI, 2.0*M_PI);
				lat_d = lat * 180.0 / M_PI;
				lon_d = lon * 180.0 / M_PI;
				if (lon_d < mid_lon) {
					if (lat_d < mid_lat) {
						which_panel = 2;
					} else {
						which_panel = 1;
					}
				} else {
					if (lat_d < mid_lat) {
						which_panel = 3;
					} else {
						which_panel = 0;
					}
				}
				octo_panels[start+which_panel].point_count += 1;
				points_inside[start+which_panel].push_back(index);
			}

			point_count = 0;
			for (int j = 0; j < 4; j++) {
				point_count += octo_panels[start+j].point_count;
			}
			if (point_count != octo_panels[i].point_count) {
				std::cout << i << " " << point_count << " " << octo_panels[i].point_count << std::endl;
				throw std::runtime_error("Error with point assignment in cubed sphere tree construction");
			}
		}
	}

	double pc[3], p1[3], p2[3], p3[3], p4[3], d1, d2, d3, d4;
	for (int i = 0; i < octo_panels.size(); i++) {
		min_lat = octo_panels[i].min_lat*M_PI/180.0;
		max_lat = octo_panels[i].max_lat*M_PI/180.0;
		min_lon = octo_panels[i].min_lon*M_PI/180.0;
		max_lon = octo_panels[i].max_lon*M_PI/180.0;
		mid_lat = 0.5*(min_lat + max_lat);
		mid_lon = 0.5*(min_lon + max_lon);
		xyz_from_lonlat(pc[0], pc[1], pc[2], mid_lon, mid_lat);
		xyz_from_lonlat(p1[0], p1[1], p1[2], min_lon, min_lat);
		xyz_from_lonlat(p2[0], p2[1], p2[2], min_lon, max_lat);
		xyz_from_lonlat(p3[0], p3[1], p3[2], max_lon, max_lat);
		xyz_from_lonlat(p4[0], p4[1], p4[2], max_lon, min_lat);
		d1 = std::acos(pc[0] * p1[0] + pc[1] * p1[1] + pc[2] * p1[2]);
		d2 = std::acos(pc[0] * p2[0] + pc[1] * p2[1] + pc[2] * p2[2]);
		d3 = std::acos(pc[0] * p3[0] + pc[1] * p3[1] + pc[2] * p3[2]);
		d4 = std::acos(pc[0] * p4[0] + pc[1] * p4[1] + pc[2] * p4[2]);
		octo_panels[i].radius = std::max(std::max(std::max(d1, d2), d3), d4);
	}

	Kokkos::resize(octo_sphere_tree, octo_panels.size());
	// Kokkos::resize(leaf_panel_point_ids, octo_panels.size(), run_config.leaf_size);
	Kokkos::resize(leaf_panel_point_ids, octo_panels.size(), 2025);
	for (int i = 0; i < octo_panels.size(); i++) {
		for (int j = 0; j < run_config.leaf_size; j++) {
			leaf_panel_point_ids(i,j) = -1;
		}
	}
	run_config.levels = -1;
	run_config.panel_count = octo_panels.size();
	for (int i = 0; i < octo_panels.size(); i++) {
		octo_sphere_tree(i) = octo_panels[i];
		// if (octo_sphere_tree(i).is_leaf) {
		for (int j = 0; j < octo_sphere_tree(i).point_count; j++) {
			leaf_panel_point_ids(i,j) = points_inside[i][j];
		}
		// }
		run_config.levels = std::max(run_config.levels, octo_sphere_tree(i).level);
	}
	run_config.cubed_sphere_level_start = (int*) malloc((run_config.levels+1) * sizeof(int));
	int lev;
	int target_lev = 0;
	for (int i = 0; i < octo_sphere_tree.extent_int(0); i++) {
		lev = octo_sphere_tree(i).level;
		if (lev == target_lev) {
			run_config.cubed_sphere_level_start[lev] = i;
			target_lev += 1;
		}
	}
}




