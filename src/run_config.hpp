#ifndef H_RUN_CONF_H
#define H_RUN_CONF_H

#include <string>

struct RunConfig {
	std::string out_path;
	int write_precision = 6; // 6 for data viz, 16 for error testing
	bool write_output = false;
	int point_count; // number of dynamics points
	int levels; // base refinement level of the cubed sphere
	int interp_degree; // interpolation degree to use
	int tracer_count;
	int end_time; // end time in seconds, 86400 = 1 day
	int delta_t; // time step in seconds
	int time_steps; // number of time steps
	std::string initial_condition; // options = sh43, 
	bool balance_ic = true; // enforce int_S vor = 0
	double fmm_theta = 0.7; // parameter for well separated threshold
	int fmm_cluster_thresh; // threshold to treat a panel as a cluster

	// mpi info
	int mpi_p; // total number of MPI ranks
	int mpi_id; // MPI id
	int oned_lb;  
	int oned_ub; // lower bound and upper bound for 1d parallelized arrays
};

#endif