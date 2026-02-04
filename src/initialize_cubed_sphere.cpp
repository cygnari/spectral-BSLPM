#include<Kokkos_Core.hpp>
#include <iostream>

#include "cubed_sphere_transforms.hpp"
// #include "cubed_sphere_transforms_impl.hpp"
#include "general_utils.hpp"
#include "initialize_cubed_sphere.hpp"
#include "run_config.hpp"
#include "initialize_cubed_sphere.hpp"
#include "general_utils_impl.hpp"

struct compute_point {
	RunConfig run_config;
	Kokkos::View<double*, Kokkos::HostSpace> xcos;
	Kokkos::View<double*, Kokkos::HostSpace> ycos;
	Kokkos::View<double*, Kokkos::HostSpace> zcos;
	Kokkos::View<double*, Kokkos::HostSpace> area;
	double small_offset;
	double dist_between;
	int points_per_side;

	compute_point(const RunConfig& run_config_, Kokkos::View<double*, Kokkos::HostSpace>& xcos_, 
					Kokkos::View<double*, Kokkos::HostSpace>& ycos_, 
					Kokkos::View<double*, Kokkos::HostSpace>& zcos_, 
					Kokkos::View<double*, Kokkos::HostSpace>& area_, 
					double small_offset_, double dist_between_, int points_per_side_) :
					run_config(run_config_), xcos(xcos_), ycos(ycos_), zcos(zcos_), 
					area(area_), dist_between(dist_between_), small_offset(small_offset_), 
					points_per_side(points_per_side_) {}

	// KOKKOS_INLINE_FUNCTION
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

void cubed_sphere_midpoints(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& xcos, 
							Kokkos::View<double*, Kokkos::HostSpace>& ycos, 
							Kokkos::View<double*, Kokkos::HostSpace>& zcos, 
							Kokkos::View<double*, Kokkos::HostSpace>& area) {
	int points_per_side = Kokkos::pow(2, run_config.levels);

	double dist_between = M_PI / (2.0*points_per_side);
	double small_offset = dist_between/2.0;

	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultHostExecutionSpace(), {0,0,0},{6,points_per_side,points_per_side}), compute_point(run_config, xcos, ycos, zcos, area, small_offset, dist_between, points_per_side));
	Kokkos::fence();
}

struct refine_cube {
	RunConfig run_config;
	Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace> cubed_sphere_panels;
	int level;

	refine_cube(const RunConfig& run_config_, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels_, int level_) :
				run_config(run_config_), cubed_sphere_panels(cubed_sphere_panels_), level(level_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		// refine panel i of prev level
		int level_start = run_config.cubed_sphere_level_start[level+1];
		int start = level_start+4*i;
		int prev_start = run_config.cubed_sphere_level_start[level];
		for (int j = 0; j < 4; j++) {
			cubed_sphere_panels(start+j).id = start+j;
			cubed_sphere_panels(start+j).face = cubed_sphere_panels(i+prev_start).face;
			cubed_sphere_panels(start+j).parent_id = i + prev_start;
			cubed_sphere_panels(start+j).level = cubed_sphere_panels(prev_start+i).level + 1;
		}
		double max_xi = cubed_sphere_panels(i+prev_start).max_xi;
		double min_xi = cubed_sphere_panels(i+prev_start).min_xi;
		double max_eta = cubed_sphere_panels(i+prev_start).max_eta;
		double min_eta = cubed_sphere_panels(i+prev_start).min_eta;
		double mid_xi = 0.5 * (max_xi + min_xi);
		double mid_eta = 0.5 * (max_eta + min_eta);

		cubed_sphere_panels(start).max_xi = max_xi;
		cubed_sphere_panels(start).min_xi = mid_xi;
		cubed_sphere_panels(start).max_eta = max_eta;
		cubed_sphere_panels(start).min_eta = mid_eta;

		cubed_sphere_panels(start+1).max_xi = mid_xi;
		cubed_sphere_panels(start+1).min_xi = min_xi;
		cubed_sphere_panels(start+1).max_eta = max_eta;
		cubed_sphere_panels(start+1).min_eta = mid_eta;

		cubed_sphere_panels(start+2).max_xi = mid_xi;
		cubed_sphere_panels(start+2).min_xi = min_xi;
		cubed_sphere_panels(start+2).max_eta = mid_eta;
		cubed_sphere_panels(start+2).min_eta = min_eta;

		cubed_sphere_panels(start+3).max_xi = max_xi;
		cubed_sphere_panels(start+3).min_xi = mid_xi;
		cubed_sphere_panels(start+3).max_eta = mid_eta;
		cubed_sphere_panels(start+3).min_eta = min_eta;

		int prev_top = cubed_sphere_panels(prev_start+i).id_top_edge - prev_start;
		int prev_left = cubed_sphere_panels(prev_start+i).id_left_edge - prev_start;
		int prev_bot = cubed_sphere_panels(prev_start+i).id_bot_edge - prev_start;
		int prev_right = cubed_sphere_panels(prev_start+i).id_right_edge - prev_start;

		cubed_sphere_panels(start).id_top_edge = level_start +4*prev_top+ 3;
		cubed_sphere_panels(start).id_left_edge = level_start+4*i+1;
		cubed_sphere_panels(start).id_bot_edge = level_start+4*i+3;
		cubed_sphere_panels(start).id_right_edge = level_start +4*prev_right+ 1;

		cubed_sphere_panels(start+1).id_top_edge = level_start +4*prev_top+ 2;
		cubed_sphere_panels(start+1).id_left_edge = level_start+4*prev_left;
		cubed_sphere_panels(start+1).id_bot_edge = level_start+4*i+2;
		cubed_sphere_panels(start+1).id_right_edge = level_start +4*i;

		cubed_sphere_panels(start+2).id_top_edge = level_start +4*i+ 1;
		cubed_sphere_panels(start+2).id_left_edge = level_start+4*prev_left+3;
		cubed_sphere_panels(start+2).id_bot_edge = level_start+4*prev_bot+1;
		cubed_sphere_panels(start+2).id_right_edge = level_start +4*i+ 3;

		cubed_sphere_panels(start+3).id_top_edge = level_start +4*i;
		cubed_sphere_panels(start+3).id_left_edge = level_start+4*i+2;
		cubed_sphere_panels(start+3).id_bot_edge = level_start+4*prev_bot;
		cubed_sphere_panels(start+3).id_right_edge = level_start +4*prev_right+ 2;

		cubed_sphere_panels(i+prev_start).child1 = start;
		cubed_sphere_panels(i+prev_start).child2 = start+1;
		cubed_sphere_panels(i+prev_start).child3 = start+2;
		cubed_sphere_panels(i+prev_start).child4 = start+3;
	}
};

struct cubed_sphere_init_2 {
	RunConfig run_config;
	Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace> cubed_sphere_panels;

	cubed_sphere_init_2(const RunConfig& run_config_, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels_) :
						run_config(run_config_), cubed_sphere_panels(cubed_sphere_panels_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		if (cubed_sphere_panels(i).level == run_config.levels - 1) {
			cubed_sphere_panels(i).is_leaf = true;
		} else {
			cubed_sphere_panels(i).is_leaf = false;
		}

		double max_xi, mid_xi, min_xi, max_eta, mid_eta, min_eta, d1, d2, d3, d4;
		max_xi = cubed_sphere_panels(i).max_xi * Kokkos::numbers::pi / 4.0; 
		min_xi = cubed_sphere_panels(i).min_xi * Kokkos::numbers::pi / 4.0; 
		max_eta = cubed_sphere_panels(i).max_eta * Kokkos::numbers::pi / 4.0; 
		min_eta = cubed_sphere_panels(i).min_eta * Kokkos::numbers::pi / 4.0; 
		mid_xi = 0.5 * (max_xi + min_xi);
		mid_eta = 0.5 * (max_eta + min_eta);

		double p1[3], p2[3], p3[3], p4[3], pc[3];
		xyz_from_xieta(max_xi, max_eta, cubed_sphere_panels(i).face, p1);
		xyz_from_xieta(max_xi, min_eta, cubed_sphere_panels(i).face, p2);
		xyz_from_xieta(min_xi, min_eta, cubed_sphere_panels(i).face, p3);
		xyz_from_xieta(min_xi, max_eta, cubed_sphere_panels(i).face, p4);
		xyz_from_xieta(mid_xi, mid_eta, cubed_sphere_panels(i).face, pc);
		d1 = gcdist(p1, pc);
		d2 = gcdist(p2, pc);
		d3 = gcdist(p3, pc);
		d4 = gcdist(p4, pc);
		cubed_sphere_panels(i).radius = Kokkos::fmax(d1, Kokkos::fmax(d2, Kokkos::fmax(d3, d4)));

		double area1, area2;
		area1 = sphere_tri_area(p1, p2, p3);
		area2 = sphere_tri_area(p1, p3, p4);
		cubed_sphere_panels(i).area = area1 + area2;
	}
};

void cubed_sphere_panels_init(const RunConfig& run_config, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels) {
	for (int i = 0; i < 6; i++) {
		cubed_sphere_panels(i).id = i;
		cubed_sphere_panels(i).face = i;
		cubed_sphere_panels(i).min_xi = -1;
		cubed_sphere_panels(i).max_xi = 1;
		cubed_sphere_panels(i).min_eta = -1;
		cubed_sphere_panels(i).max_eta = 1;
		cubed_sphere_panels(i).level = 0;
		cubed_sphere_panels(i).parent_id = -1;
	}

	for (int i = 0; i < 4; i++) {
		cubed_sphere_panels(i).id_top_edge = 4;
		cubed_sphere_panels(i).id_bot_edge = 5;
	}

	cubed_sphere_panels(0).id_left_edge = 3;
	cubed_sphere_panels(0).id_right_edge = 1;
	cubed_sphere_panels(1).id_left_edge = 0;
	cubed_sphere_panels(1).id_right_edge = 2;
	cubed_sphere_panels(2).id_left_edge = 1;
	cubed_sphere_panels(2).id_right_edge = 3;
	cubed_sphere_panels(3).id_left_edge = 2;
	cubed_sphere_panels(3).id_right_edge = 0;

	cubed_sphere_panels(4).id_top_edge=2;
	cubed_sphere_panels(4).id_left_edge=3;
	cubed_sphere_panels(4).id_bot_edge=0;
	cubed_sphere_panels(4).id_right_edge=1;

	cubed_sphere_panels(5).id_top_edge=0;
	cubed_sphere_panels(5).id_left_edge=3;
	cubed_sphere_panels(5).id_bot_edge=2;
	cubed_sphere_panels(5).id_right_edge=1;

	run_config.cubed_sphere_level_start[0] = 0;
	for (int i = 0; i < run_config.levels-1; i++) {
		run_config.cubed_sphere_level_start[i+1] = run_config.cubed_sphere_level_start[i] + 6 * pow(4, i);
	}

	for (int i = 0; i < run_config.levels-1; i++) {
		Kokkos::parallel_for(Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, 6 * pow(4, i)), refine_cube(run_config, cubed_sphere_panels, i));
		Kokkos::fence();
	}

	Kokkos::parallel_for(Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), 0, cubed_sphere_panels.extent_int(0)), cubed_sphere_init_2(run_config, cubed_sphere_panels));
	Kokkos::fence();
} 

struct compute_point_panel {
	RunConfig run_config;
	Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace> cubed_sphere_panels;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_vals;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos; 
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos; 
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area;

	compute_point_panel(const RunConfig& run_config_, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace> cubed_sphere_panels_, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_vals_, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos_,
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos_, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos_, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area_) : run_config(run_config_), interp_vals(interp_vals_),
					cubed_sphere_panels(cubed_sphere_panels_), xcos(xcos_), ycos(ycos_), zcos(zcos_), area(area_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int offset = run_config.cubed_sphere_level_start[run_config.levels-1];
		double max_xi = M_PI/4.0*cubed_sphere_panels(i).max_xi;
		double min_xi = M_PI/4.0*cubed_sphere_panels(i).min_xi;
		double max_eta = M_PI/4.0*cubed_sphere_panels(i).max_eta;
		double min_eta = M_PI/4.0*cubed_sphere_panels(i).min_eta;
		double xyz1[3], xyz2[3], xyz3[3], xyz4[3], panel_area;
		xyz_from_xieta(min_xi, min_eta, cubed_sphere_panels(i).face, xyz1);
		xyz_from_xieta(max_xi, min_eta, cubed_sphere_panels(i).face, xyz2);
		xyz_from_xieta(max_xi, max_eta, cubed_sphere_panels(i).face, xyz3);
		xyz_from_xieta(min_xi, max_eta, cubed_sphere_panels(i).face, xyz4);
		panel_area = cubed_sphere_panels(i).area;
		double xi_range, xi_offset, eta_range, eta_offset;
		xi_range = 0.5*(max_xi - min_xi), xi_offset = 0.5*(max_xi + min_xi);
		eta_range = 0.5*(max_eta - min_eta), eta_offset = 0.5*(max_eta + min_eta); 
		double xyz[3];
		for (int j = 0; j < pow(run_config.interp_degree+1, 2); j++) {
			xyz_from_xieta(xi_offset+xi_range*interp_vals(j, 0), eta_offset+eta_range*interp_vals(j, 1), cubed_sphere_panels(i).face, xyz);
			xcos(i-offset,j) = xyz[0];
			ycos(i-offset,j) = xyz[1];
			zcos(i-offset,j) = xyz[2];
			area(i-offset,j) = interp_vals(j,3) * panel_area / 4.0;
		}
	}
};

void cube_sphere_spec_points(const RunConfig& run_config, Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace>& cubed_sphere_panels, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_vals, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area) {
	Kokkos::parallel_for(Kokkos::RangePolicy(Kokkos::DefaultHostExecutionSpace(), run_config.cubed_sphere_level_start[run_config.levels-1], cubed_sphere_panels.extent_int(0)), compute_point_panel(run_config, cubed_sphere_panels, interp_vals, xcos, ycos, zcos, area));
	Kokkos::fence();
}

void cubed_sphere_2d_to_1d(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& xcos_1d, Kokkos::View<double*, Kokkos::HostSpace>& ycos_1d,
							Kokkos::View<double*, Kokkos::HostSpace>& zcos_1d, Kokkos::View<int*, Kokkos::HostSpace>& one_d_no_of_points, 
							Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& xcos, 
							Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& ycos, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& zcos) {
	int points_found = 0;
	double x, y, z;
	bool found = false;
	for (int i = 0; i < xcos.extent_int(0); i++) {
		for (int j = 0; j < xcos.extent_int(1); j++) {
			x = xcos(i,j);
			y = ycos(i,j);
			z = zcos(i,j);
			// std::cout << x << " " << y << " " << z << " " << points_found << std::endl;
			for (int k = 0; k < points_found; k++) {
				// std::cout << abs(x-xcos_1d(k)) << " " << abs(y-ycos_1d(k)) << " " << abs(z-zcos_1d(k)) << std::endl;
				if ((abs(x-xcos_1d(k)) < 1e-15) and (abs(y-ycos_1d(k)) < 1e-15) and (abs(z-zcos_1d(k)) < 1e-15)) {
					// std::cout << "found" << std::endl;
					found = true;
					one_d_no_of_points(k) += 1;
					two_d_to_1d(i,j) = k;
					break;
				}
			}
			if (not found) {
				xcos_1d(points_found) = x;
				ycos_1d(points_found) = y;
				zcos_1d(points_found) = z;
				one_d_no_of_points(points_found) += 1;
				two_d_to_1d(i,j) = points_found;
				points_found++;
				// std::cout << "not found" << std::endl;
			}
			found = false;
		}
	}
	std::cout << "points found: " << points_found << std::endl;
}

struct accumulate_vals {
	Kokkos::View<int**, Kokkos::LayoutRight> two_d_to_1d;
	Kokkos::View<double*> one_d_vals;
	Kokkos::View<double**, Kokkos::LayoutRight> vals;

	accumulate_vals(Kokkos::View<int**, Kokkos::LayoutRight>& two_d_to_1d_, Kokkos::View<double*>& one_d_vals_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& vals_) : two_d_to_1d(two_d_to_1d_), one_d_vals(one_d_vals_), vals(vals_) {}

	void operator()(const int i) const {
		int jmax = vals.extent_int(1);
		for (int j = 0; j < jmax; j++) {
			Kokkos::atomic_add(&one_d_vals(two_d_to_1d(i,j)), vals(i,j));
		}
	}
};

struct average_vals {
	Kokkos::View<int**, Kokkos::LayoutRight> two_d_to_1d;
	Kokkos::View<double*> one_d_vals;
	Kokkos::View<double**, Kokkos::LayoutRight> vals;
	Kokkos::View<int*> one_d_no_of_points;

	average_vals(Kokkos::View<int**, Kokkos::LayoutRight>& two_d_to_1d_, Kokkos::View<double*>& one_d_vals_, 
					Kokkos::View<double**, Kokkos::LayoutRight>& vals_, Kokkos::View<int*>& one_d_no_of_points_) : 
					two_d_to_1d(two_d_to_1d_), one_d_vals(one_d_vals_), vals(vals_), one_d_no_of_points(one_d_no_of_points_) {}

	void operator()(const int i) const {
		int jmax = vals.extent_int(1);
		int ind;
		for (int j = 0; j < jmax; j++) {
			ind = two_d_to_1d(i,j);
			vals(i,j) = one_d_vals(ind) / one_d_no_of_points(ind);
		}
	}
};

void unify_boundary_vals(const RunConfig& run_config, Kokkos::View<int*>& one_d_no_of_points, 
							Kokkos::View<int**, Kokkos::LayoutRight>& two_d_to_1d, Kokkos::View<double**, Kokkos::LayoutRight>& vals) {
	Kokkos::View<double*> one_d_vals ("one d vals", run_config.point_count);
	Kokkos::parallel_for(run_config.point_count, zero_out_1(one_d_vals));
	Kokkos::parallel_for(run_config.active_panel_count, accumulate_vals(two_d_to_1d, one_d_vals, vals));
	Kokkos::parallel_for(run_config.active_panel_count, average_vals(two_d_to_1d, one_d_vals, vals, one_d_no_of_points));
}

struct sol_2d_to_1d{
	Kokkos::View<double*, Kokkos::HostSpace> area_1d;
	Kokkos::View<double*, Kokkos::HostSpace> pots_1d;
	Kokkos::View<double*, Kokkos::HostSpace> soln_1d;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> area;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> pots;
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> soln;
	Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace> two_d_to_1d;

	sol_2d_to_1d(Kokkos::View<double*, Kokkos::HostSpace>& area_1d_, Kokkos::View<double*, Kokkos::HostSpace>& pots_1d_, Kokkos::View<double*, Kokkos::HostSpace>& soln_1d_, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area_, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& pots_, 
					Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& soln_, Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d_) :
					area_1d(area_1d_), pots_1d(pots_1d_), soln_1d(soln_1d_), area(area_), pots(pots_), soln(soln_), two_d_to_1d(two_d_to_1d_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		int loc = two_d_to_1d(i,j);
		Kokkos::atomic_add(&area_1d(loc), area(i,j));
		if (soln_1d(loc) == 0) {
			Kokkos::atomic_add(&soln_1d(loc), soln(i,j));
		}
		if (pots_1d(loc) == 0) {
			Kokkos::atomic_add(&pots_1d(loc), pots(i,j));
		}
	}
};

struct one_d_average {
	Kokkos::View<double*, Kokkos::HostSpace> pots_1d;
	Kokkos::View<double*, Kokkos::HostSpace> soln_1d;
	Kokkos::View<int*, Kokkos::HostSpace> one_d_no_of_points;

	one_d_average(Kokkos::View<double*, Kokkos::HostSpace>& pots_1d_, Kokkos::View<double*, Kokkos::HostSpace>& soln_1d_, Kokkos::View<int*, Kokkos::HostSpace>& one_d_no_of_points_) :
					pots_1d(pots_1d_), soln_1d(soln_1d_), one_d_no_of_points(one_d_no_of_points_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int count = one_d_no_of_points(i);
		// count = 1;
		pots_1d(i) /= count;
		soln_1d(i) /= count;
		std::cout << i << " " << count << std::endl;
	}
};

struct poisson_regularize {
	Kokkos::View<double*, Kokkos::HostSpace> pots_1d;
	Kokkos::View<double*, Kokkos::HostSpace> area_1d;
	Kokkos::View<double*, Kokkos::HostSpace> soln_1d;

	poisson_regularize(Kokkos::View<double*, Kokkos::HostSpace>& pots_1d_, Kokkos::View<double*, Kokkos::HostSpace>& area_1d_, 
						Kokkos::View<double*, Kokkos::HostSpace>& soln_1d_) : pots_1d(pots_1d_), area_1d(area_1d_), soln_1d(soln_1d_) {}

	// KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		soln_1d(i) += area_1d(i) * pots_1d(i);
	}
};

void solution_2d_to_1d(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& area_1d, Kokkos::View<double*, Kokkos::HostSpace>& pots_1d, 
						Kokkos::View<double*, Kokkos::HostSpace>& soln_1d, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& area, 
						Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& pots, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& soln, 
						Kokkos::View<int*, Kokkos::HostSpace>& one_d_no_of_points, Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d) {
	Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultHostExecutionSpace(), {0, 0}, {area.extent_int(0), area.extent_int(1)}), sol_2d_to_1d(area_1d, pots_1d, soln_1d, area, pots, soln, two_d_to_1d));
}

template struct v_2d_to_1d<Kokkos::LayoutRight>;
template struct v_2d_to_1d<Kokkos::LayoutStride>;

template void vec_2d_to_1d<Kokkos::LayoutRight>(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& vec_1d, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& vec, 
						Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d, bool add);
template void vec_2d_to_1d<Kokkos::LayoutStride>(const RunConfig& run_config, Kokkos::View<double*, Kokkos::HostSpace>& vec_1d, Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace>& vec, 
						Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>& two_d_to_1d, bool add);

