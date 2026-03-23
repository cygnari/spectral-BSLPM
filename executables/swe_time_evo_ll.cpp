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
#include "initialize_octo_sphere.hpp"
#include "initialize_cubed_sphere.hpp"
#include "interp_funcs.hpp"
#include "io_utils.hpp"
#include "mpi_utils.hpp"
#include "topography.hpp"
#include "swe_time_step_ll.hpp"

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

		// int lat_count = 180.0 / run_config.grid_spacing + 1.0;
		int lat_count = 180.0 / run_config.grid_spacing;
		int lon_count = 360.0 / run_config.grid_spacing;
		run_config.lat_count = lat_count;
		run_config.lon_count = lon_count;
		run_config.point_count = lat_count * lon_count;

		Kokkos::View<double*, Kokkos::HostSpace> lats ("lats", lat_count);
		Kokkos::View<double*, Kokkos::HostSpace> lons ("lons", lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos ("xcos", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos ("ycos", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos ("zcos", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> area ("area", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vors ("vors", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> divs ("divs", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> height ("height", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> topo ("topo", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_u ("vel_u", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_v ("vel_v", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_x ("vel_x", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_y ("vel_y", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vel_z ("vel_z", lat_count, lon_count);

		for (int i = 0; i < lat_count; i++) {
			lats(i) = -90.0 + run_config.grid_spacing * i + 0.5*run_config.grid_spacing;
		}
		for (int i = 0; i < lon_count; i++) {
			lons(i) = run_config.grid_spacing * i + 0.5*run_config.grid_spacing;
		}

		Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultHostExecutionSpace(), {0, 0}, {lat_count, lon_count}), latlon_point(lats, lons, xcos, ycos, zcos, area, run_config.grid_spacing));
		apply_topography_host_ll(run_config, lats, lons, topo);
		swe_initialize_ll(run_config, lats, lons, vors, divs, height, topo);

		double total_area = 0;

		int leaf_panel_pc = 4*(run_config.interp_degree+1)*(run_config.interp_degree+1);
		run_config.leaf_size = leaf_panel_pc;
		// Kokkos::View<OctoSpherePanel*, Kokkos::HostSpace> sphere_panels ("fmm sphere panels", 0);
		Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace> sphere_panels ("fmm sphere panels", 0);
		Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace> leaf_panel_points ("leaf panel points", 0, 0);

		// initialize_octo_sphere(run_config, xcos, ycos, zcos, sphere_panels, leaf_panel_points);
		initialize_cube_sphere_irreg_points(run_config, xcos, ycos, zcos, sphere_panels, leaf_panel_points);

		for (int i = 0; i < area.extent_int(0); i++) {
			for (int j = 0; j < area.extent_int(1); j++) {
				total_area += area(i,j);
			}
		}

		if (run_config.mpi_id == 0) {
			std::cout << "dynamics points: " << lat_count * lon_count << std::endl;
			std::cout << "tree panel count: " << run_config.panel_count << std::endl;
			std::cout << "max tree depth: " << run_config.levels << std::endl;
			std::cout << "area discrepancy from 4pi: " << total_area - 4 * M_PI << std::endl;
		}

		std::vector<double> time_vals (run_config.time_steps+1, 0);
		for (int i = 0; i < run_config.time_steps + 1; i++) {
			time_vals[i] = run_config.delta_t * i * run_config.output_freq;
		}

		// create the netcdf output
		int ncid, dims, dimids[3], lon_dimid, lat_dimid, time_dimid;
		int lat_varid, lon_varid, time_varid, uvel_id, vvel_id, vor_id, div_id, height_id, topo_id, x_id, y_id, z_id, area_id, xvel_id, yvel_id, zvel_id;
		size_t start_nc[3], count_nc[3];
		int retval;

		dims = 3;

		if (run_config.mpi_id == 0) {
			if (run_config.write_output) {
				std::string output_folder = std::to_string(run_config.lat_count) +"_" + std::to_string(run_config.lon_count) + std::string("_swe_") + run_config.initial_condition;
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
				nc_def_dim(ncid, "latitude", lat_count, &lat_dimid);
				nc_def_dim(ncid, "longitude", lon_count, &lon_dimid);
				nc_def_var(ncid, "latitude", NC_DOUBLE, 1, &lat_dimid, &lat_varid);
				nc_def_var(ncid, "longitude", NC_DOUBLE, 1, &lon_dimid, &lon_varid);

				dimids[0] = time_dimid;
				dimids[1] = lat_dimid;
				dimids[2] = lon_dimid;
				// count_nc[0] = run_config.time_steps+1;
				count_nc[0] = 1;
				count_nc[1] = lat_count;
				count_nc[2] = lon_count;
				start_nc[0] = 0; // start_nc[0] is the time step index being written
				start_nc[1] = 0;
				start_nc[2] = 0;
				nc_def_var(ncid, "vor", NC_DOUBLE, dims, dimids, &vor_id);
				nc_def_var(ncid, "div", NC_DOUBLE, dims, dimids, &div_id);
				nc_def_var(ncid, "h", NC_DOUBLE, dims, dimids, &height_id);
				nc_def_var(ncid, "topo", NC_DOUBLE, dims-1, &dimids[1], &topo_id);
				nc_def_var(ncid, "x", NC_DOUBLE, dims-1, &dimids[1], &x_id);
				nc_def_var(ncid, "y", NC_DOUBLE, dims-1, &dimids[1], &y_id);
				nc_def_var(ncid, "z", NC_DOUBLE, dims-1, &dimids[1], &z_id);
				nc_def_var(ncid, "area", NC_DOUBLE, dims-1, &dimids[1], &area_id);
				nc_put_att_text(ncid, lat_varid, "units", strlen("degrees_north"), "degrees_north");
				nc_put_att_text(ncid, lon_varid, "units", strlen("degrees_east"), "degrees_east");
				nc_put_att_text(ncid, vor_id, "units", strlen("1/s"), "1/s");
				nc_put_att_text(ncid, div_id, "units", strlen("1/s"), "1/s");
				nc_put_att_text(ncid, height_id, "units", strlen("m"), "m");
				nc_put_att_text(ncid, topo_id, "units", strlen("m"), "m");
				nc_put_att_text(ncid, vor_id, "long name", strlen("relative vorticity"), "relative vorticity");
				nc_put_att_text(ncid, div_id, "long name", strlen("fluid divergence"), "fluid divergence");
				nc_put_att_text(ncid, height_id, "long name", strlen("fluid column thickness"), "fluid column thickness");
				nc_put_att_text(ncid, topo_id, "long name", strlen("topography height"), "topography height");
				nc_put_att_text(ncid, x_id, "long name", strlen("x coordinate"), "x coordinate");
				nc_put_att_text(ncid, y_id, "long name", strlen("y coordinate"), "y coordinate");
				nc_put_att_text(ncid, z_id, "long name", strlen("z coordinate"), "z coordinate");
				nc_put_att_text(ncid, area_id, "long name", strlen("area"), "area");
				nc_def_var(ncid, "u_vel", NC_DOUBLE, dims, dimids, &uvel_id);
				nc_def_var(ncid, "v_vel", NC_DOUBLE, dims, dimids, &vvel_id);
				nc_def_var(ncid, "x_vel", NC_DOUBLE, dims, dimids, &xvel_id);
				nc_def_var(ncid, "y_vel", NC_DOUBLE, dims, dimids, &yvel_id);
				nc_def_var(ncid, "z_vel", NC_DOUBLE, dims, dimids, &zvel_id);
				nc_put_att_text(ncid, uvel_id, "units", strlen("m/s"), "m/s");
				nc_put_att_text(ncid, vvel_id, "units", strlen("m/s"), "m/s");
				nc_put_att_text(ncid, xvel_id, "units", strlen("m/s"), "m/s");
				nc_put_att_text(ncid, yvel_id, "units", strlen("m/s"), "m/s");
				nc_put_att_text(ncid, zvel_id, "units", strlen("m/s"), "m/s");
				nc_put_att_text(ncid, uvel_id, "long name", strlen("zonal velocity"), "zonal velocity");
				nc_put_att_text(ncid, vvel_id, "long name", strlen("meridional velocity"), "meridional velocity");
				nc_put_att_text(ncid, xvel_id, "long name", strlen("x velocity"), "x velocity");
				nc_put_att_text(ncid, yvel_id, "long name", strlen("y velocity"), "y velocity");
				nc_put_att_text(ncid, zvel_id, "long name", strlen("z velocity"), "z velocity");
				nc_enddef(ncid);
				nc_put_att_text(ncid, NC_GLOBAL, "initial condition", strlen(run_config.initial_condition.c_str()), run_config.initial_condition.c_str());
				nc_put_att_int(ncid, NC_GLOBAL, "point count", NC_INT, 1, &run_config.point_count);
				nc_put_att_double(ncid, NC_GLOBAL, "grid spacing", NC_DOUBLE, 1, &run_config.grid_spacing);
				nc_put_att_int(ncid, NC_GLOBAL, "time step size [s]", NC_INT, 1, &run_config.delta_t);
				nc_put_att_int(ncid, NC_GLOBAL, "end time [s]", NC_INT, 1, &run_config.end_time);
				nc_put_att_double(ncid, NC_GLOBAL, "fmm theta", NC_DOUBLE, 1, &run_config.fmm_theta);
				nc_put_att_int(ncid, NC_GLOBAL, "fmm interp degree", NC_INT, 1, &run_config.interp_degree);
				nc_put_att_int(ncid, NC_GLOBAL, "fmm tree panel count", NC_INT, 1, &run_config.panel_count);
				nc_put_var_double(ncid, lat_varid, &lats(0));
				nc_put_var_double(ncid, lon_varid, &lons(0));
				nc_put_var_double(ncid, time_varid, &time_vals[0]);
				nc_put_vara_double(ncid, vor_id, start_nc, count_nc, &vors(0,0));
				nc_put_vara_double(ncid, div_id, start_nc, count_nc, &divs(0,0));
				nc_put_vara_double(ncid, height_id, start_nc, count_nc, &height(0,0));
				nc_put_vara_double(ncid, topo_id, start_nc, &count_nc[1], &topo(0,0));
				nc_put_vara_double(ncid, x_id, start_nc, &count_nc[1], &xcos(0,0));
				nc_put_vara_double(ncid, y_id, start_nc, &count_nc[1], &ycos(0,0));
				nc_put_vara_double(ncid, z_id, start_nc, &count_nc[1], &zcos(0,0));
				nc_put_vara_double(ncid, area_id, start_nc, &count_nc[1], &area(0,0));
			}
		}

		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);
		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "initialization time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();
		Kokkos::View<interact_pair*, Kokkos::HostSpace> interaction_list ("interaction list", 0);
		// octo_tree_traversal(run_config, sphere_panels, interaction_list);
		dual_tree_traversal_irreg(run_config, sphere_panels, interaction_list);
		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);
		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "dual tree traversal time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
			std::cout << "interaction count: " << run_config.fmm_interaction_count << std::endl; 
		}

		begin = std::chrono::steady_clock::now();

		Kokkos::View<double*> d_lats ("device lats", lat_count);
		Kokkos::View<double*> d_lons ("device lons", lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_xcos ("device xcos", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_ycos ("device ycos", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_zcos ("device zcos", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_area ("device area", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_vors ("device vors", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_divs ("device divs", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_height ("device height", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_topo ("device topo", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_u ("device vel_u", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_v ("device vel_v", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_x ("device vel_x", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_y ("device vel_y", lat_count, lon_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_vel_z ("device vel_z", lat_count, lon_count);
		// Kokkos::View<OctoSpherePanel*> d_sphere_panels ("device cubed sphere panels", run_config.panel_count);
		Kokkos::View<CubedSpherePanel*> d_sphere_panels ("device cubed sphere panels", run_config.panel_count);
		Kokkos::View<int**, Kokkos::LayoutRight> d_leaf_panel_points ("device leaf panel points", run_config.panel_count, run_config.leaf_size);
		// Kokkos::View<int**, Kokkos::LayoutRight> d_leaf_panel_points ("device leaf panel points", run_config.panel_count, 2025);
		Kokkos::View<interact_pair*> d_interaction_list ("device interaction list", run_config.fmm_interaction_count);

		Kokkos::deep_copy(d_xcos, xcos);
		Kokkos::deep_copy(d_ycos, ycos);
		Kokkos::deep_copy(d_zcos, zcos);
		Kokkos::deep_copy(d_area, area);
		Kokkos::deep_copy(d_vors, vors);
		Kokkos::deep_copy(d_divs, divs);
		Kokkos::deep_copy(d_height, height);
		Kokkos::deep_copy(d_topo, topo);
		Kokkos::deep_copy(d_sphere_panels, sphere_panels);
		Kokkos::deep_copy(d_interaction_list, interaction_list);
		Kokkos::deep_copy(d_leaf_panel_points, leaf_panel_points);

		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "host to device communication time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();
		for (int t = 0; t < run_config.time_steps; t++) {
			std::cout << "time step: " << t << std::endl;
			swe_back_rk4_step_ll(run_config, d_xcos, d_ycos, d_zcos, d_area, d_vors, d_divs, d_height, d_topo, d_sphere_panels, d_interaction_list, d_leaf_panel_points, d_vel_x, d_vel_y, d_vel_z, run_config.delta_t*t);

			Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {d_vel_u.extent_int(0), d_vel_u.extent_int(1)}), zero_out(d_vel_u));
			Kokkos::parallel_for(Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), {0, 0}, {d_vel_v.extent_int(0), d_vel_v.extent_int(1)}), zero_out(d_vel_v));
			Kokkos::parallel_for(run_config.lat_count, xyz_vel_to_uv_vel(d_xcos, d_ycos, d_zcos, d_vel_x, d_vel_y, d_vel_z, d_vel_u, d_vel_v));

			Kokkos::deep_copy(vel_u, d_vel_u);
			Kokkos::deep_copy(vel_v, d_vel_v);	
			Kokkos::deep_copy(vel_x, d_vel_x);
			Kokkos::deep_copy(vel_y, d_vel_y);
			Kokkos::deep_copy(vel_z, d_vel_z);	
			Kokkos::deep_copy(vors, d_vors);
			Kokkos::deep_copy(divs, d_divs);
			Kokkos::deep_copy(height, d_height);
			start_nc[0] = t;
			nc_put_vara_double(ncid, uvel_id, start_nc, count_nc, &vel_u(0,0));
			nc_put_vara_double(ncid, vvel_id, start_nc, count_nc, &vel_v(0,0));
			nc_put_vara_double(ncid, xvel_id, start_nc, count_nc, &vel_x(0,0));
			nc_put_vara_double(ncid, yvel_id, start_nc, count_nc, &vel_y(0,0));
			nc_put_vara_double(ncid, zvel_id, start_nc, count_nc, &vel_z(0,0));
			start_nc[0] = t+1;
			nc_put_vara_double(ncid, vor_id, start_nc, count_nc, &vors(0,0));
			nc_put_vara_double(ncid, div_id, start_nc, count_nc, &divs(0,0));
			nc_put_vara_double(ncid, height_id, start_nc, count_nc, &height(0,0));
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
