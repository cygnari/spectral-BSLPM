#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "specBSLPM-config.h"
#include "direct_sum.hpp"
#include "fmm_funcs.hpp"
#include "general_utils.hpp"
#include "initial_conditions.hpp"
#include "initialize_cubed_sphere.hpp"
#include "interp_funcs.hpp"
#include "io_utils.hpp"
#include "mpi_utils.hpp"

#include "fmm_interactions/inverse_poisson.hpp"

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
		
		run_config.point_count = run_config.active_panel_count * pow(run_config.interp_degree, 2) + 2;

		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> interp_vals ("interp vals", pow(run_config.interp_degree+1, 2), 4);

		interp_init(run_config, interp_vals);

		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xcos ("xcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> ycos ("ycos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> zcos ("zcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> area ("area", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> pots ("pots", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> soln ("soln", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));

		cube_sphere_spec_points(run_config, cubed_sphere_panels, interp_vals, xcos, ycos, zcos, area);

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

		poisson_initialize(run_config, xcos, ycos, zcos, pots);

		// for (int i = 0; i < pots.extent_int(0); i++) {
		// 	for (int j = 0; j < pots.extent_int(1); j++) {
		// 		std::cout << pots(i,j) << " " << area(i,j) << std::endl;
		// 	}
		// }

		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);
		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "initialization time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
			// for (int i = 0; i < xcos.extent_int(0); i++) {
			// 	for (int j = 0; j < xcos.extent_int(1); j++) {
			// 		std::cout << xcos(i,j) << " " << ycos(i,j) << " " << zcos(i, j) << std::endl;
			// 	}	
			// }
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
			// for (int i = 0; i < run_config.fmm_interaction_count; i++) {
			// 	std::cout << interaction_list(i).target_panel << " " << interaction_list(i).source_panel << " " << interaction_list(i).interact_type << std::endl;
			// }
		}
		begin = std::chrono::steady_clock::now();

		// move to device
		Kokkos::View<double**, Kokkos::LayoutRight> d_xcos("device xcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_ycos("device ycos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_zcos("device zcos", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_area("device area", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_pots("device pots", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<double**, Kokkos::LayoutRight> d_soln("device soln", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));
		Kokkos::View<CubedSpherePanel*> d_cubed_sphere_panels ("device cubed sphere panels", run_config.panel_count);
		Kokkos::View<interact_pair*> d_interaction_list("device interaction list", run_config.fmm_interaction_count);
		Kokkos::View<double**, Kokkos::LayoutRight> d_interp_vals ("device interp vals", pow(run_config.interp_degree+1, 2), 4);

		Kokkos::deep_copy(d_xcos, xcos);
		Kokkos::deep_copy(d_ycos, ycos);
		Kokkos::deep_copy(d_zcos, zcos);
		Kokkos::deep_copy(d_area, area);
		Kokkos::deep_copy(d_pots, pots);
		Kokkos::deep_copy(d_cubed_sphere_panels, cubed_sphere_panels);
		Kokkos::deep_copy(d_interaction_list, interaction_list);
		Kokkos::deep_copy(d_interp_vals, interp_vals);

		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "host to device communication time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();

		Kokkos::View<double**, Kokkos::LayoutRight> d_proxy_source_weights("device proxy source weights", run_config.panel_count, pow(run_config.interp_degree+1, 2));

		// for (int i = 0; i < d_pots.extent_int(0); i++) {
		// 	for (int j = 0; j < d_pots.extent_int(1); j++) {
		// 		std::cout << d_pots(i,j) * d_area(i,j) << std::endl;
		// 	}
		// }

		upward_pass(run_config, d_interp_vals, d_cubed_sphere_panels, d_area, d_pots, d_proxy_source_weights);

		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "upward pass time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();
		// std::cout << d_proxy_source_weights(0,0) << std::endl;
		// for (int i = 0; i < d_proxy_source_weights.extent_int(0); i++) {
		// 	for (int j = 0; j < d_proxy_source_weights.extent_int(1); j++) {
		// 		std::cout << d_proxy_source_weights(i,j) << std::endl;
		// 	}
		// }

		Kokkos::View<double**, Kokkos::LayoutRight> d_proxy_target_potentials("device proxy target potentials", run_config.panel_count, pow(run_config.interp_degree+1, 2));

		// for (int i = 0; i < d_proxy_target_potentials.extent_int(0); i++) {
		// for (int j = 0; j < d_proxy_target_potentials.extent_int(1); j++) {
		// 	std::cout << d_proxy_target_potentials(0,j) << std::endl;
		// }
		// }
		// for (int i = 0; i < d_proxy_target_potentials.extent_int(0); i++) {
		// 	for (int j = 0; j < d_proxy_target_potentials.extent_int(1); j++) {
		// 		std::cout << d_proxy_target_potentials(i,j) << std::endl;
		// 	}
		// }

		poisson_fmm_interactions(run_config, d_proxy_target_potentials, d_proxy_source_weights, d_interaction_list, d_cubed_sphere_panels, d_interp_vals);

		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "interaction time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();
		// std::cout << d_proxy_target_potentials(0,0) << std::endl;
		// for (int i = 0; i < d_proxy_target_potentials.extent_int(0); i++) {
		// 	for (int j = 0; j < d_proxy_target_potentials.extent_int(1); j++) {
		// 		std::cout << d_proxy_target_potentials(i,j) << std::endl;
		// 	}
		// }
		// for (int j = 0; j < d_proxy_target_potentials.extent_int(1); j++) {
		// 	std::cout << d_proxy_target_potentials(0,j) << std::endl;
		// }

		downward_pass(run_config, d_interp_vals, d_cubed_sphere_panels, d_proxy_target_potentials, d_soln);

		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "downward pass time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();
		// std::cout << d_soln(0,0) << std::endl;

		// back to host
		Kokkos::deep_copy(soln, d_soln);
		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);
		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "device to host communication time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
			// std::cout << soln(0,0) << std::endl;
		}
		begin = std::chrono::steady_clock::now();
		MPI_Allreduce(MPI_IN_PLACE, &soln(0,0), run_config.active_panel_count * pow(run_config.interp_degree+1,2), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "global reduction time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}
		begin = std::chrono::steady_clock::now();
		// int i = 0;
		// std::cout << cubed_sphere_panels(i+6).id_right_edge << " " << soln(i,0) << " " << soln(cubed_sphere_panels(i+6).id_right_edge,20)<< std::endl;
		// std::cout << soln(i,0)-soln(cubed_sphere_panels(i+6).id_right_edge,20)<< std::endl;

		// for (int i = 0; i < pots_1d.extent_int(0); i++) {
		// 	std::cout << xcos_1d(i) << " " << ycos_1d(i) << " " << zcos_1d(i) << " " << pots_1d(i) << " " << one_d_no_of_points(i) << std::endl;
		// }

		if (run_config.write_output) {
			Kokkos::View<double*, Kokkos::HostSpace> xcos_1d ("1d x cos", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> ycos_1d ("1d y cos", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> zcos_1d ("1d z cos", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> area_1d ("1d areas", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> pots_1d ("1d pots", run_config.point_count);
			Kokkos::View<double*, Kokkos::HostSpace> soln_1d ("1d soln", run_config.point_count);

			Kokkos::View<int*, Kokkos::HostSpace> one_d_no_of_points ("number of points collapsed", run_config.point_count);

			Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace> two_d_to_1d ("two d to 1d map", run_config.active_panel_count, pow(run_config.interp_degree+1, 2));

			cubed_sphere_2d_to_1d(run_config, xcos_1d, ycos_1d, zcos_1d, one_d_no_of_points, two_d_to_1d, xcos, ycos, zcos);
			solution_2d_to_1d(run_config, area_1d, pots_1d, soln_1d, area, pots, soln, one_d_no_of_points, two_d_to_1d);

			if (run_config.mpi_id == 0) {
				std::string output_folder = std::to_string(run_config.point_count) + "_"+ std::to_string(run_config.levels) +"_" + std::to_string(run_config.interp_degree) + std::string("_fmm_inv_lap_") + run_config.initial_condition;
				std::string command = std::string("python ") + NAMELIST_DIR + std::string("initialize.py ") + run_config.out_path + "/" + output_folder;
				system(command.c_str());
				std::string outpath = run_config.out_path + "/" + output_folder + "/";
				write_state(xcos_1d, outpath, "x.csv", run_config.write_precision);
				write_state(ycos_1d, outpath, "y.csv", run_config.write_precision);
				write_state(zcos_1d, outpath, "z.csv", run_config.write_precision);
				write_state(area_1d, outpath, "a.csv", run_config.write_precision);
				write_state(pots_1d, outpath, "p.csv", run_config.write_precision);
				write_state(soln_1d, outpath, "s.csv", run_config.write_precision);
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