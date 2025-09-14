#ifndef H_DIRECT_SUM_H
#define H_DIRECT_SUM_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"

void direct_sum_inv_lap(const RunConfig& run_config, Kokkos::View<double*> xcos, Kokkos::View<double*> ycos, Kokkos::View<double*> zcos, Kokkos::View<double*> area, Kokkos::View<double*> pots, Kokkos::View<double*> soln);

#endif
