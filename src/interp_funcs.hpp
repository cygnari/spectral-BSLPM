#ifndef H_INTERP_H
#define H_INTERP_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"

void interp_init(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& interp_info);

#endif
