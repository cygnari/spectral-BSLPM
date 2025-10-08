#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "specBSLPM-config.h"
#include "deriv_funcs.hpp"
#include "direct_sum.hpp"
#include "fmm_funcs.hpp"
#include "general_utils.hpp"
#include "initial_conditions.hpp"
#include "initialize_cubed_sphere.hpp"
#include "interp_funcs.hpp"
#include "io_utils.hpp"
#include "mpi_utils.hpp"

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

	Kokkos::initialize(argc, argv);
	{
		Kokkos::View<CubedSpherePanel*, Kokkos::HostSpace> cubed_sphere_panels ("cubed sphere panels", run_config.panel_count);
		
		cubed_sphere_panels_init(run_config, cubed_sphere_panels);
		std::cout << "leaf level panels: " << run_config.active_panel_count << std::endl;
		run_config.point_count = run_config.active_panel_count * pow(run_config.interp_degree, 2) + 2;

		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> interp_vals ("interp vals", pow(run_config.interp_degree+1, 2), 4);

		interp_init(run_config, interp_vals);

		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos ("xcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos ("ycos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos ("zcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> area ("area", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> pots ("pots", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> grad_x ("x grad", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> grad_y ("y grad", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> grad_z ("z grad", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));

		cube_sphere_spec_points(run_config, cubed_sphere_panels, interp_vals, xcos, ycos, zcos, area);

		double total_area = 0;

		for (int i = 0; i < area.extent_int(0); i++) {
			for (int j = 0; j < area.extent_int(1); j++) {
				total_area += area(i,j);
			}
		}

		if (run_config.mpi_id == 0) {
			std::cout << "area discrepancy from 4pi: " << total_area - 4 * M_PI << std::endl;
		}

		poisson_initialize(run_config, xcos, ycos, zcos, pots);

		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);

		// move to device
		Kokkos::View<double**, Kokkos::LayoutRight> d_xcos("device xcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_ycos("device ycos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_zcos("device zcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_pots("device pots", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_grad_x ("device x grad", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_grad_y ("device y grad", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_grad_z ("device z grad", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<CubedSpherePanel*> d_cubed_sphere_panels ("device cubed sphere panels", run_config.panel_count);
		Kokkos::View<interact_pair*> d_interaction_list("device interaction list", run_config.fmm_interaction_count);

		Kokkos::deep_copy(d_xcos, xcos);
		Kokkos::deep_copy(d_ycos, ycos);
		Kokkos::deep_copy(d_zcos, zcos);
		Kokkos::deep_copy(d_pots, pots);
		Kokkos::deep_copy(d_cubed_sphere_panels, cubed_sphere_panels);
		
		xyz_gradient(run_config, d_grad_x, d_grad_y, d_grad_z, d_pots, d_cubed_sphere_panels);

		// back to host
		// Kokkos::deep_copy(soln, d_soln);
		Kokkos::deep_copy(grad_x, d_grad_x);
		Kokkos::deep_copy(grad_y, d_grad_y);
		Kokkos::deep_copy(grad_z, d_grad_z);
		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);
		// end = std::chrono::steady_clock::now();
		// if (run_config.mpi_id == 0) {
		// 	std::cout << "device to host communication time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		// 	// std::cout << soln(0,0) << std::endl;
		// }
		// begin = std::chrono::steady_clock::now();
		MPI_Allreduce(MPI_IN_PLACE, &grad_x(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &grad_y(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &grad_z(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		// end = std::chrono::steady_clock::now();
		// if (run_config.mpi_id == 0) {
		// 	std::cout << "global reduction time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		// }
		// begin = std::chrono::steady_clock::now();

		if (run_config.write_output) {
			Kokkos::View<double*, Kokkos::HostSpace> xcos_1d ("1d x cos", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> ycos_1d ("1d y cos", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> zcos_1d ("1d z cos", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> area_1d ("1d areas", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> pots_1d ("1d pots", run_config.point_count);
			// Kokkos::View<double*, Kokkos::HostSpace> soln_1d ("1d soln", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> grad_x_1d ("1d grad x", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> grad_y_1d ("1d grad y", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> grad_z_1d ("1d grad z", run_config.point_count);

			Kokkos::View<int*, Kokkos::HostSpace> one_d_no_of_points ("number of points collapsed", run_config.point_count);

			Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace> two_d_to_1d ("two d to 1d map", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));

			cubed_sphere_2d_to_1d(run_config, xcos_1d, ycos_1d, zcos_1d, one_d_no_of_points, two_d_to_1d, xcos, ycos, zcos);
			solution_2d_to_1d(run_config, area_1d, pots_1d, grad_x_1d, area, pots, grad_x, one_d_no_of_points, two_d_to_1d);
			solution_2d_to_1d(run_config, area_1d, pots_1d, grad_y_1d, area, pots, grad_y, one_d_no_of_points, two_d_to_1d);
			solution_2d_to_1d(run_config, area_1d, pots_1d, grad_z_1d, area, pots, grad_z, one_d_no_of_points, two_d_to_1d);

			if (run_config.mpi_id == 0) {
				std::string output_folder = std::to_string(run_config.point_count) + "_"+ std::to_string(run_config.levels) +"_" + std::to_string(run_config.interp_degree) + std::string("_grad_") + run_config.initial_condition;
				std::string command = std::string("python ") + NAMELIST_DIR + std::string("initialize.py ") + run_config.out_path + "/" + output_folder;
				system(command.c_str());
				std::string outpath = run_config.out_path + "/" + output_folder + "/";
				write_state(xcos_1d, outpath, "x.csv", run_config.write_precision);
				write_state(ycos_1d, outpath, "y.csv", run_config.write_precision);
				write_state(zcos_1d, outpath, "z.csv", run_config.write_precision);
				// write_state(area_1d, outpath, "a.csv", run_config.write_precision);
				write_state(pots_1d, outpath, "p.csv", run_config.write_precision);
				write_state(grad_x_1d, outpath, "grad_x.csv", run_config.write_precision);
				write_state(grad_y_1d, outpath, "grad_y.csv", run_config.write_precision);
				write_state(grad_z_1d, outpath, "grad_z.csv", run_config.write_precision);
			}
		}
		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "write output time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();
	}
	Kokkos::finalize();

	MPI_Finalize();

	return 0;
}