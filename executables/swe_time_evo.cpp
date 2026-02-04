#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <netcdf.h>

#include "specBSLPM-config.h"
#include "direct_sum.hpp"
#include "fmm_funcs.hpp"
#include "general_utils.hpp"
#include "general_utils_impl.hpp"
#include "initial_conditions.hpp"
#include "initialize_cubed_sphere.hpp"
#include "interp_funcs.hpp"
#include "io_utils.hpp"
#include "mpi_utils.hpp"
#include "swe_time_step.hpp"
#include "swe_time_step_2.hpp"
#include "topography.hpp"

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int P, ID;
	MPI_Comm_size(MPI_COMM_WORLD, &P);
	MPI_Comm_rank(MPI_COMM_WORLD, &ID);

	RunConfig run_config;
	read_run_config(std::string(NAMELIST_DIR) + std::string("namelist.txt"), run_config);

	run_config.mpi_p = P;
	run_config.mpi_id = ID;

	bounds_determine_1d(run_config, P, ID);

	std::chrono::steady_clock::time_point begin, end;
	begin = std::chrono::steady_clock::now();

	Kokkos::initialize(argc, argv); {
		if (run_config.mpi_id == 0) {
			std::cout << "mpi ranks: " << P << std::endl;
			std::cout << "kokkos num threads: " << Kokkos::num_threads() << ", Kokkos num devices: " << Kokkos::num_devices() << std::endl;
		}

		Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace> cubed_sphere_panels ("cubed sphere panels", run_config.panel_count);
		
		cubed_sphere_panels_init(run_config, cubed_sphere_panels);
		
		run_config.point_count = run_config.active_panel_count * pow(run_config.interp_degree, 2) + 2;

		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> interp_vals ("interp vals", pow(run_config.interp_degree+1, 2), 4);

		interp_init(run_config, interp_vals);

		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos ("xcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos ("ycos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos ("zcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> area ("area", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vors ("vors", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> divs ("divs", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> height ("height", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> height_lap ("height_lap", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_x ("vel_x", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_y ("vel_y", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_z ("vel_z", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_u ("vel_u", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_v ("vel_v", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace> passive_tracers ("passive tracers", run_config.active_panel_count, pow(run_config.interp_degree+1, 2), run_config.tracer_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> topo ("topography", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));

		std::map<std::string, Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace>> individual_tracers; 
		for (int i = 0; i < run_config.tracer_count; i++) {
			individual_tracers[run_config.tracers[i]] = Kokkos::subview(passive_tracers, Kokkos::ALL, Kokkos::ALL, i);
		}

		cube_sphere_spec_points(run_config, cubed_sphere_panels, interp_vals, xcos, ycos, zcos, area);
		apply_topography_2(run_config, xcos, ycos, zcos, vel_x, vel_y, vel_z, vel_u, topo); // vel_x/y/z/u are all 0 at this point

		double total_area = 0;

		for (int i = 0; i < area.extent_int(0); i++) {
			for (int j = 0; j < area.extent_int(1); j++) {
				total_area += area(i,j);
			}
		}

		if (run_config.mpi_id == 0) {
			std::cout << "leaf level panels: " << run_config.active_panel_count << std::endl;
			std::cout << "area discrepancy from 4pi: " << total_area - 4 * M_PI << std::endl;
		}

		swe_initialize(run_config, xcos, ycos, zcos, vors, divs, height, area);
		tracer_initialize(run_config, xcos, ycos, zcos, vors, passive_tracers);

		// initialize output file
		double output_grid_spacing = std::pow(2.0, std::floor(std::log2(std::sqrt(4*M_PI / run_config.point_count) * 180.0 / M_PI)));
		int lat_count = 180.0 / output_grid_spacing + 1.0;
		int lon_count = 360.0 / output_grid_spacing;

		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vor_out ("vorticity output", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> div_out ("divergence output", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> height_out ("height output", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> uvel_out ("u velocity output", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vvel_out ("v velocity output", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> tracer_out ("passive tracer output", lat_count, lon_count);
		Kokkos::View<double*, Kokkos::HostSpace> lat_vals ("output lats", lat_count);
		Kokkos::View<double*, Kokkos::HostSpace> lon_vals ("output lons", lon_count);
		Kokkos::View<double*, Kokkos::HostSpace> xcos_1d ("1d x cos", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> ycos_1d ("1d y cos", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> zcos_1d ("1d z cos", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> area_1d ("1d areas", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> lats_1d ("1d lats", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> lons_1d ("1d lons", run_config.point_count);
		Kokkos::View<int*, Kokkos::HostSpace> one_d_no_of_points ("number of points collapsed", run_config.point_count);
		Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace> two_d_to_1d ("two d to 1d map", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double*, Kokkos::HostSpace> time_vals ("time steps", run_config.time_steps + 1);
		Kokkos::View<int*, Kokkos::HostSpace> point_indices ("point indices", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> one_d_vec_out ("one d vec to write output", run_config.point_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> topo_out ("topography output", lat_count, lon_count);

		cubed_sphere_2d_to_1d(run_config, xcos_1d, ycos_1d, zcos_1d, one_d_no_of_points, two_d_to_1d, xcos, ycos, zcos);

		int ncid, dims;
		for (int i = 0; i < run_config.time_steps + 1; i++) {
			time_vals(i) = run_config.delta_t * i * run_config.output_freq;
		}
		int dimids[3], dimids_no_t[2]; // lat, lon, time
		int lon_dimid, lat_dimid, time_dimid, lat_varid, lon_varid, time_varid, uvel_id, vvel_id, vor_id, div_id, height_id, group_id, vel_x_id, vel_y_id, vel_z_id, height_l_id, topo_id;
		int point_dimid, point_varid, xco_varid, yco_varid, zco_varid, area_varid;
		size_t start_nc[3], count_nc[3], count_nc_no_t[2]; // where to write output
		int tracer_varids[run_config.tracer_count];
		int retval;

		if (run_config.mpi_id == 0) {
			if (run_config.write_output) {
				std::string output_folder = std::to_string(run_config.levels) +"_" + std::to_string(run_config.interp_degree) + std::string("_swe_") + run_config.initial_condition;
				std::string command = std::string("python ") + NAMELIST_DIR + std::string("initialize.py ") + run_config.out_path + "/" + output_folder;
				system(command.c_str());
				std::string outpath = run_config.out_path + "/" + output_folder + "/output.nc";
				retval = nc_create(outpath.c_str(), NC_NETCDF4, &ncid);
				if (retval != 0) {
					std::cout << nc_strerror(retval) << std::endl;
					throw std::runtime_error("NC file creation issue");
				}
				nc_def_dim(ncid, "time", run_config.time_steps / run_config.output_freq + 1, &time_dimid);
				nc_def_var(ncid, "time", NC_INT, 1, &time_dimid, &time_varid);
				nc_put_att_text(ncid, time_varid, "units", strlen("s"), "s");
				if (run_config.interp_output) {
					dims = 3;
					nc_def_dim(ncid, "latitude", lat_count, &lat_dimid);
					nc_def_dim(ncid, "longitude", lon_count, &lon_dimid);
					nc_def_var(ncid, "latitude", NC_DOUBLE, 1, &lat_dimid, &lat_varid);
					nc_def_var(ncid, "longitude", NC_DOUBLE, 1, &lon_dimid, &lon_varid);
					
					dimids[0] = time_dimid;
					dimids[1] = lat_dimid;
					dimids[2] = lon_dimid;
					dimids_no_t[0] = lat_dimid;
					dimids_no_t[1] = lon_dimid;
					
					count_nc[0] = 1;
					count_nc[1] = lat_count;
					count_nc[2] = lon_count;
					start_nc[0] = 0; // start_nc[0] is the time step index being written
					start_nc[1] = 0;
					start_nc[2] = 0;
					count_nc_no_t[0] = lat_count;
					count_nc_no_t[1] = lon_count;
				} else {
					dims = 2;
					nc_def_dim(ncid, "point_index", run_config.point_count, &point_dimid);
					nc_def_var(ncid, "point_index", NC_INT, 1, &point_dimid, &point_varid);
					dimids[0] = time_dimid;
					dimids[1] = point_dimid;
					dimids_no_t[0] = point_dimid;
					nc_def_var(ncid, "x", NC_DOUBLE, 1, &point_dimid, &xco_varid);
					nc_def_var(ncid, "y", NC_DOUBLE, 1, &point_dimid, &yco_varid);
					nc_def_var(ncid, "z", NC_DOUBLE, 1, &point_dimid, &zco_varid);
					nc_def_var(ncid, "area", NC_DOUBLE, 1, &point_dimid, &area_varid);
					nc_def_var(ncid, "latitude", NC_DOUBLE, 1, &point_dimid, &lat_varid);
					nc_def_var(ncid, "longitude", NC_DOUBLE, 1, &point_dimid, &lon_varid);
					count_nc[0] = 1;
					count_nc[1] = run_config.point_count;
					start_nc[0] = 0;
					start_nc[1] = 0;
					count_nc_no_t[0] = run_config.point_count;
				}
				nc_def_var(ncid, "vor", NC_DOUBLE, dims, dimids, &vor_id);
				nc_def_var(ncid, "div", NC_DOUBLE, dims, dimids, &div_id);
				nc_def_var(ncid, "h", NC_DOUBLE, dims, dimids, &height_id);
				nc_def_var(ncid, "topo", NC_DOUBLE, dims-1, dimids_no_t, &topo_id);
				// nc_def_var(ncid, "h_lap", NC_DOUBLE, dims, dimids, &height_l_id);
				for (int i = 0; i < run_config.tracer_count; i++) {
					nc_def_var(ncid, std::string("tracer_" + run_config.tracers[i]).c_str(), NC_DOUBLE, dims, dimids, &tracer_varids[i]);
				}
				nc_put_att_text(ncid, lat_varid, "units", strlen("degrees_north"), "degrees_north");
				nc_put_att_text(ncid, lon_varid, "units", strlen("degrees_east"), "degrees_east");
				nc_put_att_text(ncid, vor_id, "units", strlen("1/s"), "1/s");
				nc_put_att_text(ncid, div_id, "units", strlen("1/s"), "1/s");
				nc_put_att_text(ncid, height_id, "units", strlen("m"), "m");
				nc_put_att_text(ncid, topo_id, "units", strlen("m"), "m");
				nc_put_att_text(ncid, vor_id, "long name", strlen("relative vorticity"), "relative vorticity");
				nc_put_att_text(ncid, div_id, "long name", strlen("fluid divergence"), "fluid divergence");
				nc_put_att_text(ncid, height_id, "long name", strlen("free surface height"), "free surface height");
				nc_put_att_text(ncid, topo_id, "long name", strlen("topography height"), "topography height");
				if (run_config.output_vel) {
					nc_def_var(ncid, "u_vel", NC_DOUBLE, dims, dimids, &uvel_id);
					nc_def_var(ncid, "v_vel", NC_DOUBLE, dims, dimids, &vvel_id);
					nc_def_var(ncid, "vel_x", NC_DOUBLE, dims, dimids, &vel_x_id);
					nc_def_var(ncid, "vel_y", NC_DOUBLE, dims, dimids, &vel_y_id);
					nc_def_var(ncid, "vel_z", NC_DOUBLE, dims, dimids, &vel_z_id);
					nc_put_att_text(ncid, uvel_id, "units", strlen("m/s"), "m/s");
					nc_put_att_text(ncid, vvel_id, "units", strlen("m/s"), "m/s");
					nc_put_att_text(ncid, vel_x_id, "units", strlen("m/s"), "m/s");
					nc_put_att_text(ncid, vel_y_id, "units", strlen("m/s"), "m/s");
					nc_put_att_text(ncid, vel_z_id, "units", strlen("m/s"), "m/s");
					nc_put_att_text(ncid, uvel_id, "long name", strlen("zonal velocity"), "zonal velocity");
					nc_put_att_text(ncid, vvel_id, "long name", strlen("meridional velocity"), "meridional velocity");
					nc_put_att_text(ncid, vel_x_id, "long name", strlen("x velocity"), "x velocity");
					nc_put_att_text(ncid, vel_y_id, "long name", strlen("x velocity"), "y velocity");
					nc_put_att_text(ncid, vel_z_id, "long name", strlen("x velocity"), "z velocity");
				}
				nc_enddef(ncid);
				nc_put_att_text(ncid, NC_GLOBAL, "initial condition", strlen(run_config.initial_condition.c_str()), run_config.initial_condition.c_str());
				nc_put_att_int(ncid, NC_GLOBAL, "point count", NC_INT, 1, &run_config.point_count);
				nc_put_att_int(ncid, NC_GLOBAL, "interp degree", NC_INT, 1, &run_config.interp_degree);
				nc_put_att_int(ncid, NC_GLOBAL, "tree levels", NC_INT, 1, &run_config.levels);
				nc_put_att_int(ncid, NC_GLOBAL, "panel count", NC_INT, 1, &run_config.panel_count);
				nc_put_att_int(ncid, NC_GLOBAL, "active panel count", NC_INT, 1, &run_config.active_panel_count);
				nc_put_att_int(ncid, NC_GLOBAL, "time step size [s]", NC_INT, 1, &run_config.delta_t);
				nc_put_att_int(ncid, NC_GLOBAL, "end time", NC_INT, 1, &run_config.end_time);
				nc_put_att_double(ncid, NC_GLOBAL, "fmm theta", NC_DOUBLE, 1, &run_config.fmm_theta);
				if (run_config.interp_output) {
					for (int i = 0; i < lat_count; i++) {
						lat_vals(i) = -90.0 + output_grid_spacing * i;
					}
					for (int i = 0; i < lon_count; i++) {
						lon_vals(i) = output_grid_spacing * i;
					}
					nc_put_var_double(ncid, lat_varid, &lat_vals(0));
					nc_put_var_double(ncid, lon_varid, &lon_vals(0));
					interp_to_latlon(run_config, vors, cubed_sphere_panels, vor_out, lat_vals, lon_vals);
					nc_put_vara_double(ncid, vor_id, start_nc, count_nc, &vor_out(0,0));
					interp_to_latlon(run_config, divs, cubed_sphere_panels, div_out, lat_vals, lon_vals);
					nc_put_vara_double(ncid, div_id, start_nc, count_nc, &div_out(0,0));
					interp_to_latlon(run_config, height, cubed_sphere_panels, height_out, lat_vals, lon_vals);
					nc_put_vara_double(ncid, height_id, start_nc, count_nc, &height_out(0,0));
					for (int i = 0; i < run_config.tracer_count; i++) {
						interp_to_latlon(run_config, individual_tracers[run_config.tracers[i]], cubed_sphere_panels, tracer_out, lat_vals, lon_vals);
						nc_put_vara_double(ncid, tracer_varids[i], start_nc, count_nc, &tracer_out(0,0));
					}
					interp_to_latlon(run_config, topo, cubed_sphere_panels, topo_out, lat_vals, lon_vals);
					nc_put_vara_double(ncid, topo_id, start_nc, count_nc_no_t, &topo_out(0,0));
				} else {
					vec_2d_to_1d<Kokkos::LayoutRight>(run_config, area_1d, area, two_d_to_1d, true);			
					nc_put_var_double(ncid, xco_varid, &xcos_1d(0));
					nc_put_var_double(ncid, yco_varid, &ycos_1d(0));
					nc_put_var_double(ncid, zco_varid, &zcos_1d(0));
					nc_put_var_double(ncid, area_varid, &area_1d(0));
					nc_put_att_text(ncid, area_varid, "long name", strlen("area on a sphere of radius 1"), "area on a sphere of radius 1");
					for (int i = 0; i < run_config.point_count; i++) {
						point_indices(i) = i;
					}
					nc_put_var_int(ncid, point_varid, &point_indices(0));
					vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, vors, two_d_to_1d, false);
					nc_put_vara_double(ncid, vor_id, start_nc, count_nc, &one_d_vec_out(0));
					vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, divs, two_d_to_1d, false);
					nc_put_vara_double(ncid, div_id, start_nc, count_nc, &one_d_vec_out(0));
					vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, height, two_d_to_1d, false);
					nc_put_vara_double(ncid, height_id, start_nc, count_nc, &one_d_vec_out(0));
					vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, topo, two_d_to_1d, false);
					nc_put_vara_double(ncid, topo_id, start_nc, count_nc_no_t, &one_d_vec_out(0));
					for (int i = 0; i < run_config.point_count; i++) {
						lats_1d(i) = 90.0 - acos(zcos_1d(i)) * 180.0 / M_PI;
						lons_1d(i) = 180.0 / M_PI * atan2(ycos_1d(i), xcos_1d(i));
					}
					nc_put_var_double(ncid, lat_varid, &lats_1d(0));
					nc_put_var_double(ncid, lon_varid, &lons_1d(0));
					for (int i = 0; i < run_config.tracer_count; i++) {
						vec_2d_to_1d<Kokkos::LayoutStride>(run_config, one_d_vec_out, individual_tracers[run_config.tracers[i]], two_d_to_1d, false);
						nc_put_vara_double(ncid, tracer_varids[i], start_nc, count_nc, &one_d_vec_out(0));
					}
				}
				nc_put_var_double(ncid, time_varid, &time_vals(0));
			}
		}

		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);
		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "initialization time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();

		Kokkos::View<interact_pair*, Kokkos::HostSpace> interaction_list ("interaction list", 1);
		dual_tree_traversal(run_config, cubed_sphere_panels, interaction_list);

		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);
		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "dual tree traversal time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
			std::cout << "interaction count: " << run_config.fmm_interaction_count << std::endl; 
		}
		begin = std::chrono::steady_clock::now();

		Kokkos::View<double**, Kokkos::LayoutRight> d_xcos("device xcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_ycos("device ycos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_zcos("device zcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_area("device area", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_vors("device vors", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_divs("device divs", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_height("device heights", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_height_lap("device height laps", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_x("device vel x", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_y("device vel y", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_z("device vel z", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_u("device vel u", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_v("device vel v", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<CubedSpherePanel*> d_cubed_sphere_panels ("device cubed sphere panels", run_config.panel_count);
		Kokkos::View<interact_pair*> d_interaction_list("device interaction list", run_config.fmm_interaction_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_interp_vals ("device interp vals", pow(run_config.interp_degree+1, 2), 4);
		Kokkos::View<double***, Kokkos::LayoutRight> d_passive_tracers("device passive tracers", run_config.active_panel_count, pow(run_config.interp_degree+1, 2), run_config.tracer_count);
		Kokkos::View<int*> d_one_d_no_of_points ("device number of points collapsed", run_config.point_count);
		Kokkos::View<int**, Kokkos::LayoutRight> d_two_d_to_1d ("device two d to 1d map", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));

		Kokkos::deep_copy(d_xcos, xcos);
		Kokkos::deep_copy(d_ycos, ycos);
		Kokkos::deep_copy(d_zcos, zcos);
		Kokkos::deep_copy(d_area, area);
		Kokkos::deep_copy(d_vors, vors);
		Kokkos::deep_copy(d_divs, divs);
		Kokkos::deep_copy(d_height, height);
		Kokkos::deep_copy(d_cubed_sphere_panels, cubed_sphere_panels);
		Kokkos::deep_copy(d_interaction_list, interaction_list);
		Kokkos::deep_copy(d_interp_vals, interp_vals);
		Kokkos::deep_copy(d_passive_tracers, passive_tracers);
		Kokkos::deep_copy(d_one_d_no_of_points, one_d_no_of_points);
		Kokkos::deep_copy(d_two_d_to_1d, two_d_to_1d);

		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "host to device communication time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();

		Kokkos::View<double**, Kokkos::LayoutRight> d_proxy_source_weights("device proxy source weights", run_config.panel_count, pow(run_config.interp_degree+1, 2));

		int offset = run_config.panel_count - run_config.active_panel_count;

		for (int t = 0; t < run_config.time_steps; t++) {
			std::cout << "time step: " << t << std::endl;
			swe_back_rk4_step(run_config, d_xcos, d_ycos, d_zcos, d_area, d_vors, d_divs, d_height, d_interp_vals, d_cubed_sphere_panels, d_interaction_list, d_vel_x, d_vel_y, d_vel_z, d_passive_tracers, d_height_lap, t * run_config.delta_t);
			unify_boundary_vals(run_config, d_one_d_no_of_points, d_two_d_to_1d, d_vors);
			unify_boundary_vals(run_config, d_one_d_no_of_points, d_two_d_to_1d, d_divs);
			unify_boundary_vals(run_config, d_one_d_no_of_points, d_two_d_to_1d, d_height);

			if (run_config.write_output) {
				if (t % run_config.output_freq == run_config.output_freq-1) {
					Kokkos::parallel_for(run_config.active_panel_count, xyz_vel_to_uv_vel(d_xcos, d_ycos, d_zcos, d_vel_x, d_vel_y, d_vel_z, d_vel_u, d_vel_v));
					Kokkos::deep_copy(vel_u, d_vel_u);
					Kokkos::deep_copy(vel_v, d_vel_v);	
					Kokkos::deep_copy(vel_x, d_vel_x);
					Kokkos::deep_copy(vel_y, d_vel_y);
					Kokkos::deep_copy(vel_z, d_vel_z);	
					// Kokkos::deep_copy(height_lap, d_height_lap);	
					Kokkos::deep_copy(vors, d_vors);
					Kokkos::deep_copy(divs, d_divs);
					Kokkos::deep_copy(height, d_height);
					Kokkos::deep_copy(passive_tracers, d_passive_tracers);

					if (run_config.output_vel) {
						start_nc[0] = t;
						if (run_config.interp_output) {
							interp_to_latlon(run_config, vel_u, cubed_sphere_panels, uvel_out, lat_vals, lon_vals);
							nc_put_vara_double(ncid, uvel_id, start_nc, count_nc, &uvel_out(0,0));
							interp_to_latlon(run_config, vel_v, cubed_sphere_panels, vvel_out, lat_vals, lon_vals);
							nc_put_vara_double(ncid, vvel_id, start_nc, count_nc, &vvel_out(0,0));
							interp_to_latlon(run_config, vel_x, cubed_sphere_panels, vvel_out, lat_vals, lon_vals);
							nc_put_vara_double(ncid, vel_x_id, start_nc, count_nc, &vvel_out(0,0));
							interp_to_latlon(run_config, vel_y, cubed_sphere_panels, vvel_out, lat_vals, lon_vals);
							nc_put_vara_double(ncid, vel_y_id, start_nc, count_nc, &vvel_out(0,0));
							interp_to_latlon(run_config, vel_z, cubed_sphere_panels, vvel_out, lat_vals, lon_vals);
							nc_put_vara_double(ncid, vel_z_id, start_nc, count_nc, &vvel_out(0,0));
							
						} else {
							vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, vel_u, two_d_to_1d, false);
							nc_put_vara_double(ncid, uvel_id, start_nc, count_nc, &one_d_vec_out(0));
							vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, vel_v, two_d_to_1d, false);
							nc_put_vara_double(ncid, vvel_id, start_nc, count_nc, &one_d_vec_out(0));
							vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, vel_x, two_d_to_1d, false);
							nc_put_vara_double(ncid, vel_x_id, start_nc, count_nc, &one_d_vec_out(0));
							vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, vel_y, two_d_to_1d, false);
							nc_put_vara_double(ncid, vel_y_id, start_nc, count_nc, &one_d_vec_out(0));
							vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, vel_z, two_d_to_1d, false);
							nc_put_vara_double(ncid, vel_z_id, start_nc, count_nc, &one_d_vec_out(0));
						}
					}
					
					start_nc[0] = (t+1)/run_config.output_freq;
					if (run_config.interp_output) {
						interp_to_latlon(run_config, vors, cubed_sphere_panels, vor_out, lat_vals, lon_vals);
						nc_put_vara_double(ncid, vor_id, start_nc, count_nc, &vor_out(0,0));
						interp_to_latlon(run_config, divs, cubed_sphere_panels, div_out, lat_vals, lon_vals);
						nc_put_vara_double(ncid, div_id, start_nc, count_nc, &div_out(0,0));
						interp_to_latlon(run_config, height, cubed_sphere_panels, height_out, lat_vals, lon_vals);
						nc_put_vara_double(ncid, height_id, start_nc, count_nc, &height_out(0,0));
						for (int i = 0; i < run_config.tracer_count; i++) {
							interp_to_latlon(run_config, individual_tracers[run_config.tracers[i]], cubed_sphere_panels, tracer_out, lat_vals, lon_vals);
							nc_put_vara_double(ncid, tracer_varids[i], start_nc, count_nc, &tracer_out(0,0));
						}
					} else {
						vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, vors, two_d_to_1d, false);
						nc_put_vara_double(ncid, vor_id, start_nc, count_nc, &one_d_vec_out(0));
						vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, divs, two_d_to_1d, false);
						nc_put_vara_double(ncid, div_id, start_nc, count_nc, &one_d_vec_out(0));
						vec_2d_to_1d<Kokkos::LayoutRight>(run_config, one_d_vec_out, height, two_d_to_1d, false);
						nc_put_vara_double(ncid, height_id, start_nc, count_nc, &one_d_vec_out(0));
						for (int i = 0; i < run_config.tracer_count; i++) {
							vec_2d_to_1d<Kokkos::LayoutStride>(run_config, one_d_vec_out, individual_tracers[run_config.tracers[i]], two_d_to_1d, false);
							nc_put_vara_double(ncid, tracer_varids[i], start_nc, count_nc, &one_d_vec_out(0));
						}
					}
				}	
			}
		}

		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);

		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "integration time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		if (run_config.write_output) {
			if (run_config.mpi_id == 0) {
				nc_close(ncid);
			}
		}
	}
	Kokkos::finalize();
	MPI_Finalize();

	return 0;
}