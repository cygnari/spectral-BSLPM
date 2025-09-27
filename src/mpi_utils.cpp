#include <vector>
#include <cassert>
#include "run_config.hpp"
#define assertm(exp, msg) assert(((void)msg, exp))

void bounds_determine_1d(RunConfig& run_information, const int P, const int ID) {
	// 1d parallel layout
	std::vector<int> particles(P, int(run_information.point_count / P));
	std::vector<int> lb(P, 0);
	std::vector<int> ub(P, 0);
	int total = P * int(run_information.point_count / P);
	int gap = run_information.point_count - total;
	for (int i = 1; i < gap + 1; i++) {
		particles[i] += 1;
	}
	total = 0;
	for (int i = 0; i < P; i++) {
		total += particles[i];
	}

	assertm(total == run_information.point_count, "Particle count not correct");

	ub[0] = particles[0];
	for (int i = 1; i < P; i++) {
		lb[i] = ub[i - 1];
		ub[i] = lb[i] + particles[i];
	}
	run_information.oned_lb = lb[ID];
	run_information.oned_ub = ub[ID];
}