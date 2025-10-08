#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include<Kokkos_Core.hpp>
#include <mpi.h>

#include "specBSLPM-config.h"
#include "general_utils.hpp"
#include "cubed_sphere_transforms_impl.hpp"

struct test_struct {
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
    std::cout << i << " " << j << std::endl;
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
    // std::cout << i << std::endl;
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
    // Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left,Kokkos::Iterate::Left>>({0, 10}, {10, 20}), test_struct());
    // Kokkos::parallel_for(size, array_fill(array));
    // Kokkos::parallel_reduce(Kokkos::RangePolicy(0, 2), sumstruct(array), sum);
    Kokkos::fence();
    end = std::chrono::steady_clock::now();

    if (ID == 0) {
      std::cout << "runtime: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
      std::cout << sum << std::endl;
    }
  }
  double lon_comp, colat_comp, x_comp, y_comp, z_comp, xi, eta, x, y, z, X, Y, C, D, delta, xyz[3];
  int face;
  face = 5;
  xi = 0.2;
  eta = 0.2;
  X = tan(xi);
  Y = tan(eta);
  xyz_from_xieta(xi, eta, face, xyz);
  x = xyz[0];
  y = xyz[1];
  z = xyz[2];
  loncolatvec_from_xietavec(lon_comp, colat_comp, 1, 0.5, face, xi, eta);
  std::cout << lon_comp << " " << colat_comp << std::endl;
  xyzvec_from_loncolatvec(x_comp, y_comp, z_comp, 0.5, 1, x, y, z);
  std::cout << x << " " << y << " " << z << std::endl;
  std::cout << x_comp << " " << y_comp << " " << z_comp << std::endl;
  
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}