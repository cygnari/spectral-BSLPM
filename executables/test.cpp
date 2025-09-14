#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include<Kokkos_Core.hpp>
#include <mpi.h>

#include "specBSLPM-config.h"
#include "general_utils.hpp"

struct test_struct {
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const {
    std::cout << i << " " << j << " " << k << std::endl;
  }
};

struct array_fill {
  Kokkos::View<double*> x;
  array_fill (Kokkos::View<double*> x_) : x(x_) {}

  KOKKOS_INLINE_FUNCTION 
  void operator()(const int i) const {
    x(i) = 1.0/pow(i+1,3);
  }
};

struct sumstruct {
  Kokkos::View<double*> x;
  sumstruct (Kokkos::View<double*> x_) : x(x_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, double& lsum) const {
    lsum += x(i);
    std::cout << i << std::endl;
  }
};

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int P, ID;
  MPI_Status status;
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &ID);
  std::chrono::steady_clock::time_point begin, end;
  Kokkos::initialize(argc,argv);
  {
    int num;
    int size = 1e8;
    num = Kokkos::num_threads();
    std::cout << P << ", " << ID << ", " << num << std::endl; 
    double sum = 0;
    begin = std::chrono::steady_clock::now();
    // Kokkos::View<double*> array ("array1", size);
    Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0, 0}, {2, 3, 5}), test_struct());
    // Kokkos::parallel_for(size, array_fill(array));
    // Kokkos::parallel_reduce(Kokkos::RangePolicy(0, 2), sumstruct(array), sum);
    Kokkos::fence();
    end = std::chrono::steady_clock::now();

    if (ID == 0) {
      std::cout << "runtime: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
      std::cout << sum << std::endl;
    }
  }
  
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}