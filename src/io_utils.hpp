#ifndef H_IO_UTIL_H
#define H_IO_UTIL_H

#include <string>
#include <Kokkos_Core.hpp>

#include "run_config.hpp"

void read_run_config(const std::string file_name, RunConfig& run_config);

void write_state(Kokkos::View<double*, Kokkos::HostSpace> &data, const std::string path, const std::string additional, const int prec);

// void write_state(Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> &data, const std::string path, const std::string additional, const int prec);

#endif
