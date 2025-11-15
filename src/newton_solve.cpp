#ifndef H_NEWTON_H
#define H_NEWTON_H

#include <Kokkos_Core.hpp>
#include <KokkosBatched_Gesv.hpp>

#include "run_config.hpp"
#include "deriv_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "newton_solve_impl.hpp"

void newton_solve(const RunConfig& run_config, Kokkos::View<double**, Kokkos::LayoutRight>& xcos, Kokkos::View<double**, Kokkos::LayoutRight>& ycos, 
				Kokkos::View<double**, Kokkos::LayoutRight>& zcos, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x, 
				Kokkos::View<double**, Kokkos::LayoutRight>& vel_y, Kokkos::View<double**, Kokkos::LayoutRight>& vel_z, 
				Kokkos::View<double**, Kokkos::LayoutRight>& dep_x, Kokkos::View<double**, Kokkos::LayoutRight>& dep_y, 
				Kokkos::View<double**, Kokkos::LayoutRight>& dep_z, Kokkos::View<CubedSpherePanel*> cubed_sphere_panels) {

	// computes the jacobian
	Kokkos::View<double**[3][3], Kokkos::LayoutRight> vel_jac ("velocity jacobian matrix", xcos.extent_int(0), xcos.extent_int(1), 3, 3);
	Kokkos::View<double**, Kokkos::LayoutRight> jac11 = Kokkos::subview(vel_jac, Kokkos::ALL(), Kokkos::ALL(), 0, 0);
	Kokkos::View<double**, Kokkos::LayoutRight> jac12 = Kokkos::subview(vel_jac, Kokkos::ALL(), Kokkos::ALL(), 0, 1);
	Kokkos::View<double**, Kokkos::LayoutRight> jac13 = Kokkos::subview(vel_jac, Kokkos::ALL(), Kokkos::ALL(), 0, 2);
	Kokkos::View<double**, Kokkos::LayoutRight> jac21 = Kokkos::subview(vel_jac, Kokkos::ALL(), Kokkos::ALL(), 1, 0);
	Kokkos::View<double**, Kokkos::LayoutRight> jac22 = Kokkos::subview(vel_jac, Kokkos::ALL(), Kokkos::ALL(), 1, 1);
	Kokkos::View<double**, Kokkos::LayoutRight> jac23 = Kokkos::subview(vel_jac, Kokkos::ALL(), Kokkos::ALL(), 1, 2);
	Kokkos::View<double**, Kokkos::LayoutRight> jac31 = Kokkos::subview(vel_jac, Kokkos::ALL(), Kokkos::ALL(), 2, 0);
	Kokkos::View<double**, Kokkos::LayoutRight> jac32 = Kokkos::subview(vel_jac, Kokkos::ALL(), Kokkos::ALL(), 2, 1);
	Kokkos::View<double**, Kokkos::LayoutRight> jac33 = Kokkos::subview(vel_jac, Kokkos::ALL(), Kokkos::ALL(), 2, 2);

	xyz_gradient(run_config, jac11, jac12, jac13, vel_x, cubed_sphere_panels);
	xyz_gradient(run_config, jac21, jac22, jac23, vel_y, cubed_sphere_panels);
	xyz_gradient(run_config, jac31, jac32, jac33, vel_z, cubed_sphere_panels);

	// computes the departure points for each (x,y,z) point using the (vel_x, vel_y, vel_z)
	int offset = cubed_sphere_panels.extent_int(0) - run_config.active_panel_count;
	Kokkos::parallel_for(xcos.extent_int(0), departure_points(
		xcos, ycos, zcos, vel_x, vel_y, vel_z, dep_x, dep_y, dep_z, cubed_sphere_panels, vel_jac, run_config.interp_degree, offset, run_config.delta_t));
}

#endif
