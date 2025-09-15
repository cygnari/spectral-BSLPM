#ifndef H_MPI_UTIL_H
#define H_MPI_UTIL_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"

void bounds_determine_1d(RunConfig& run_config, const int P, const int ID);

template <typename T> void sync_updates(Kokkos::View<T*, Kokkos::HostSpace()> &vals, const int P, const int ID,
                  const MPI_Win *win, MPI_Datatype type,
                  MPI_Comm mpi_communicator = MPI_COMM_WORLD) {
  MPI_Barrier(mpi_communicator);
  MPI_Win_fence(0, *win);
  if (ID != 0) {
    MPI_Accumulate(&vals(0), vals.extent_int(0), type, 0, 0, vals.extent_int(0), type, MPI_SUM, *win);
  }
  MPI_Win_fence(0, *win);
  if (ID != 0) {
    MPI_Get(&vals(0), vals.extent_int(0), type, 0, 0, vals.extent_int(0), type, *win);
  }
  MPI_Win_fence(0, *win);
  MPI_Barrier(mpi_communicator);
}

#endif
