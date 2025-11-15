#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "deriv_funcs.hpp"
#include "initialize_cubed_sphere.hpp"

void newton_solve(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
				Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, 
				Kokkos::View<double**, Kokkos::LayoutRight>& dep_x, Kokkos::View<double**, Kokkos::LayoutRight>& dep_y, 
				Kokkos::View<double**, Kokkos::LayoutRight>& dep_z, Kokkos::View<CubedSpherePanel*> cubed_sphere_panels);