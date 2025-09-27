#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "specBSLPM-config.h"
#include "direct_sum.hpp"
#include "general_utils.hpp"
#include "initial_conditions.hpp"
#include "initialize_cubed_sphere.hpp"
#include "io_utils.hpp"
#include "mpi_utils.hpp"

struct view_print {
	Kokkos::View<double*> array;

	view_print(Kokkos::View<double*> array_) : array(array_ ) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		std::cout << array(i) << std::endl;
	}
};

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
		int num_kokkos_threads = Kokkos::num_threads();

		Kokkos::View<double*, Kokkos::HostSpace> xcos ("x coordinates", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> ycos ("y coordinates", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> zcos ("z coordinates", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> area ("point areas", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> pots ("potentials", run_config.point_count);
		Kokkos::View<double*, Kokkos::HostSpace> soln ("solution", run_config.point_count);

		cubed_sphere_midpoints(run_config, xcos, ycos, zcos, area);

		double total_area = 0.0;

		for(int i = 0; i < run_config.point_count; i++) {
			total_area += area(i);
		}

		if (run_config.mpi_id == 0) {
			std::cout << "area discrepancy from 4pi: " << total_area - 4 * M_PI << std::endl;
			std::cout << "point count: " << run_config.point_count << std::endl;
		}

		poisson_initialize(run_config, xcos, ycos, zcos, pots);

		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);
		end = std::chrono::steady_clock::now();
		if (run_config.mpi_id == 0) {
			std::cout << "initialization time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}

		// before this is on host
		
		begin = std::chrono::steady_clock::now();

		// move to device if available

		Kokkos::View<double*> d_xcos = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace::memory_space(), xcos);
		Kokkos::View<double*> d_ycos = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace::memory_space(), ycos);
		Kokkos::View<double*> d_zcos = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace::memory_space(), zcos);
		Kokkos::View<double*> d_area = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace::memory_space(), area);
		Kokkos::View<double*> d_pots = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace::memory_space(), pots);
		Kokkos::deep_copy(d_xcos, xcos);
		Kokkos::deep_copy(d_ycos, ycos);
		Kokkos::deep_copy(d_zcos, zcos);
		Kokkos::deep_copy(d_area, area);
		Kokkos::deep_copy(d_pots, pots);
		Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> d_soln ("solution", run_config.point_count);

		direct_sum_inv_lap(run_config, d_xcos, d_ycos, d_zcos, d_area, d_pots, d_soln); // direct sum

		Kokkos::fence();

		// move back to host
		Kokkos::deep_copy(soln, d_soln);

		Kokkos::fence();
		MPI_Barrier(MPI_COMM_WORLD);
		end = std::chrono::steady_clock::now();

		if (run_config.mpi_id == 0) {
			std::cout << "integration time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		}

		if (run_config.mpi_id == 0) {
			if (run_config.write_output) {
				std::string output_folder = std::to_string(run_config.point_count) + std::string("_inv_lap_") + run_config.initial_condition;
				std::string command = std::string("python ") + NAMELIST_DIR + std::string("initialize.py ") + run_config.out_path + "/" + output_folder;
				system(command.c_str());
				std::string outpath = run_config.out_path + "/" + output_folder + "/";
				write_state(xcos, outpath, "x.csv", run_config.write_precision);
				write_state(ycos, outpath, "y.csv", run_config.write_precision);
				write_state(zcos, outpath, "z.csv", run_config.write_precision);
				write_state(area, outpath, "a.csv", run_config.write_precision);
				write_state(pots, outpath, "p.csv", run_config.write_precision);
				write_state(soln, outpath, "s.csv", run_config.write_precision);
			}
		}
	}
	Kokkos::finalize();

	MPI_Finalize();

	return 0;
}