#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include <netcdf.h>

#include "run_config.hpp"

void read_run_config(const std::string file_name, RunConfig& run_config) {
	std::ifstream config_file(file_name);
	if (config_file.fail()) {
	    std::cout << "namelist file at " << file_name << std::endl;
	    throw std::runtime_error("namelist not found");
	}
	std::string line, word1, word2;

	while (true) {
	  getline(config_file, line);
	  std::stringstream str1(line);
	  getline(str1, word1, '=');
	  getline(str1, word2);
	  if (word1 == "out_path") {
	  	run_config.out_path = word2;
	  } else if (word1 == "write_output") {
	    if (stoi(word2) == 1) {
	      run_config.write_output = true;
	    }
	  } else if (word1 == "write_precision") {
	    run_config.write_precision = stoi(word2);
	  } else if (word1 == "levels") {
	    run_config.levels = stoi(word2);
	  } else if (word1 == "interp_degree") {
	  	run_config.interp_degree = stoi(word2);
	  	if (run_config.interp_degree > 10) {
	  		run_config.interp_degree = 10;
	  	}
	  } else if (word1 == "theta") {
	    run_config.fmm_theta = stod(word2);
	  } else if (word1 == "initial_condition") {
	    run_config.initial_condition = word2;
	  } else if (word1 == "balance_ic") {
	    if (stoi(word2) == 0) {
	      run_config.balance_ic = false;
	    }
	  } else if (word1 == "end_time") {
	    run_config.end_time = stoi(word2);
	  } else if (word1 == "delta_t") {
	    run_config.delta_t = stoi(word2);
	  } else if (word1 == "tracer_count") {
	  	run_config.tracer_count = stoi(word2);
	  } else if (word1 == "omega") {
	  	run_config.omega = stod(word2);
	  } else if (word1 == "interp_output") {
	  	if (stoi(word2) == 0) {
	  		run_config.interp_output = false;
	  	}
	  } else {
	    run_config.fmm_cluster_thresh = 4 * pow(run_config.interp_degree + 1, 2);
	    run_config.time_steps = run_config.end_time / run_config.delta_t;
	    run_config.point_count = 6 * pow(4, run_config.levels);
	    run_config.panel_count = 2 * (pow(4, run_config.levels) - 1);
	    run_config.active_panel_count = 6 * pow(4, run_config.levels - 1);
	    run_config.cubed_sphere_level_start = (int*) malloc((run_config.levels) * sizeof(int));
	    return;
	  }
	}
}

void write_state(Kokkos::View<double*, Kokkos::HostSpace> &data, const std::string path, const std::string additional, const int prec) {
  std::ofstream write_out(path + additional, std::ofstream::out | std::ofstream::trunc);
  for (int i = 0; i < data.extent_int(0); i++) { // write out state
    write_out << std::setprecision(prec) << data(i) << "\n";
  }
  write_out.close();
}
