#ifndef H_INIT_CUBE_SPHERE_H
#define H_INIT_CUBE_SPHERE_H

#include<Kokkos_Core.hpp>

#include "run_config.hpp"

void cubed_sphere_midpoints(const RunConfig& run_config, Kokkos::View<double*>& xcos, Kokkos::View<double*>& ycos, Kokkos::View<double*>& zcos, Kokkos::View<double*>& area);

#endif
