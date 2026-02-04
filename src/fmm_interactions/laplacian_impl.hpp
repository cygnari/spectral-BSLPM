#ifndef H_FMM_LAP_IMPL_H
#define H_FMM_LAP_IMPL_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "general_utils_impl.hpp"

struct lap_panel_interaction {
    Kokkos::View<double**, Kokkos::LayoutRight> disp_x;
    Kokkos::View<double**, Kokkos::LayoutRight> disp_y;
    Kokkos::View<double**, Kokkos::LayoutRight> disp_z;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots;
    Kokkos::View<double**, Kokkos::LayoutRight> source_vals;
    Kokkos::View<double**, Kokkos::LayoutRight> interp_vals;
    Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
    Kokkos::View<interact_pair*> interaction_list;
    int offset;
    double kernel_eps;

    lap_panel_interaction(Kokkos::View<double**, Kokkos::LayoutRight>& disp_x_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& disp_z_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_,
                                Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals_, Kokkos::View<CubedSpherePanel*> cubed_sphere_panels_, 
                                Kokkos::View<interact_pair*>& interaction_list_, int offset_, double kernel_eps_) : disp_x(disp_x_), disp_y(disp_y_), disp_z(disp_z_), target_pots(target_pots_), source_vals(source_vals_), 
                                interp_vals(interp_vals_), cubed_sphere_panels(cubed_sphere_panels_), interaction_list(interaction_list_), offset(offset_), kernel_eps(kernel_eps_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        int target_panel = interaction_list(i).target_panel;
        int source_panel = interaction_list(i).source_panel;
        double xi_off_t, xi_scale_t, xi_off_s, xi_scale_s, eta_off_t, eta_scale_t, eta_off_s, eta_scale_s;
        xi_off_t = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(target_panel).max_xi + cubed_sphere_panels(target_panel).min_xi);
        xi_scale_t = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(target_panel).max_xi - cubed_sphere_panels(target_panel).min_xi);
        xi_off_s = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(source_panel).max_xi + cubed_sphere_panels(source_panel).min_xi);
        xi_scale_s = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(source_panel).max_xi - cubed_sphere_panels(source_panel).min_xi);

        eta_off_t = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(target_panel).max_eta + cubed_sphere_panels(target_panel).min_eta);
        eta_scale_t = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(target_panel).max_eta - cubed_sphere_panels(target_panel).min_eta);
        eta_off_s = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(source_panel).max_eta + cubed_sphere_panels(source_panel).min_eta);
        eta_scale_s = Kokkos::numbers::pi/4*0.5*(cubed_sphere_panels(source_panel).max_eta - cubed_sphere_panels(source_panel).min_eta);

        // double eps = kernel_eps;
        double eps;
        eps = 0.1;
        double xi_s, eta_s, xi_t, eta_t, xyz_t[3], xyz_s[3], dp, gfcomp, x_s, y_s, z_s, dp2, h1, h2;
        for (int j = 0; j < interp_vals.extent_int(0); j++) { // target loop
            xi_t = interp_vals(j,0) * xi_scale_t + xi_off_t;
            eta_t = interp_vals(j,1) * eta_scale_t + eta_off_t;
            xyz_from_xieta(xi_t, eta_t, cubed_sphere_panels(target_panel).face, xyz_t);
            for (int k = 0; k < interp_vals.extent(0); k++) { // source loop
                xi_s = interp_vals(k,0) * xi_scale_s + xi_off_s;
                eta_s = interp_vals(k,1) * eta_scale_s + eta_off_s;
                xyz_from_xieta(xi_s, eta_s, cubed_sphere_panels(source_panel).face, xyz_s);
                x_s = xyz_s[0] + disp_x(source_panel, k);
                y_s = xyz_s[1] + disp_y(source_panel, k);
                z_s = xyz_s[2] + disp_z(source_panel, k);
                project_to_sphere(x_s, y_s, z_s);
                dp = xyz_t[0] * x_s + xyz_t[1] * y_s + xyz_t[2] * z_s;
                dp2 = dp*dp;

                h1 = Kokkos::exp(-1.0/eps) * Kokkos::pow(Kokkos::numbers::pi, -1.5)*(2.0*dp*(-1.0+3.0*eps+dp2));
                h2 = (1.0+Kokkos::erf(dp/eps))*Kokkos::exp((dp2-1)/eps)/(Kokkos::sqrt(eps)*Kokkos::numbers::pi)*(2.0*eps*eps+2.0*dp2*(dp2-1.0)+eps*(7.0*dp2-1.0));
                // Kokkos::atomic_add(&target_pots(target_panel, j), (h1+h2)/Kokkos::pow(eps,2.5)*(source_vals(target_panel,j) - source_vals(source_panel,k)));
                Kokkos::atomic_add(&target_pots(target_panel, j), (h1+h2)/(Kokkos::pow(eps,2.5))*source_vals(source_panel,k));
            }
        }
    }
};

#endif
