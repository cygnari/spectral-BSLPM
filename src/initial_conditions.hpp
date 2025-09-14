#ifndef H_INITIAL_CONDITIONS_H
#define H_INITIAL_CONDITIONS_H

#include "Kokkos_Core.hpp"
#include "run_config.hpp"

void poisson_initialize(const RunConfig& run_config, Kokkos::View<double*>& xcos, Kokkos::View<double*>& ycos, Kokkos::View<double*>& zcos, Kokkos::View<double*>& potential);

#endif