#ifndef H_FMM_SWE_IMPL_H
#define H_FMM_SWE_IMPL_H

#include <Kokkos_Core.hpp>

#include "run_config.hpp"
#include "fmm_funcs.hpp"
#include "initialize_cubed_sphere.hpp"
#include "initialize_octo_sphere.hpp"
#include "general_utils_impl.hpp"

struct swe_vel_panel_interaction {
    Kokkos::View<double**, Kokkos::LayoutRight> disp_x;
    Kokkos::View<double**, Kokkos::LayoutRight> disp_y;
    Kokkos::View<double**, Kokkos::LayoutRight> disp_z;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_1;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_2;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_3;
    Kokkos::View<double**, Kokkos::LayoutRight> source_vals_vor;
    Kokkos::View<double**, Kokkos::LayoutRight> source_vals_div;
    Kokkos::View<double**, Kokkos::LayoutRight> interp_vals;
    Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
    Kokkos::View<interact_pair*> interaction_list;
    int offset;
    double kernel_eps;

    swe_vel_panel_interaction(Kokkos::View<double**, Kokkos::LayoutRight>& disp_x_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& disp_z_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3_, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_vor_, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_div_,
                                Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals_, Kokkos::View<CubedSpherePanel*> cubed_sphere_panels_, 
                                Kokkos::View<interact_pair*>& interaction_list_, int offset_, double kernel_eps_) : disp_x(disp_x_), disp_y(disp_y_), disp_z(disp_z_), target_pots_1(target_pots_1_), target_pots_2(target_pots_2_), 
                                target_pots_3(target_pots_3_), source_vals_vor(source_vals_vor_), source_vals_div(source_vals_div_), interp_vals(interp_vals_), cubed_sphere_panels(cubed_sphere_panels_), 
                                interaction_list(interaction_list_), offset(offset_), kernel_eps(kernel_eps_) {}

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

        double xi_s, eta_s, xi_t, eta_t, xyz_t[3], xyz_s[3], dp, gfcomp, x_s, y_s, z_s, gfcomp1, gfcomp2;
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
                // if (dp < 1.0 - 1e-15) { // constant is -1/(4pi)
                gfcomp = -0.07957747154594767 / (1.0 - dp + kernel_eps);
                gfcomp1 = gfcomp * source_vals_vor(source_panel, k);
                gfcomp2 = gfcomp * source_vals_div(source_panel, k);
                Kokkos::atomic_add(&target_pots_1(target_panel, j), gfcomp1*(xyz_t[1]*z_s - xyz_t[2]*y_s));
                Kokkos::atomic_add(&target_pots_2(target_panel, j), gfcomp1*(xyz_t[2]*x_s - xyz_t[0]*z_s));
                Kokkos::atomic_add(&target_pots_3(target_panel, j), gfcomp1*(xyz_t[0]*y_s - xyz_t[1]*x_s));
                Kokkos::atomic_add(&target_pots_1(target_panel, j), gfcomp2*(x_s - xyz_t[0] * dp));
                Kokkos::atomic_add(&target_pots_2(target_panel, j), gfcomp2*(y_s - xyz_t[1] * dp));
                Kokkos::atomic_add(&target_pots_3(target_panel, j), gfcomp2*(z_s - xyz_t[2] * dp));
                // } 
            }
        }
    }
};

struct swe_vel_panel_interaction_2 {
    Kokkos::View<double**, Kokkos::LayoutRight> disp_x;
    Kokkos::View<double**, Kokkos::LayoutRight> disp_y;
    Kokkos::View<double**, Kokkos::LayoutRight> disp_z;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_1;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_2;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_3;
    Kokkos::View<double**, Kokkos::LayoutRight> source_vals_vor;
    Kokkos::View<double**, Kokkos::LayoutRight> source_vals_div;
    Kokkos::View<double**, Kokkos::LayoutRight> interp_vals;
    Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
    Kokkos::View<interact_pair*> interaction_list;
    int offset;
    double kernel_eps;

    swe_vel_panel_interaction_2(Kokkos::View<double**, Kokkos::LayoutRight>& disp_x_, Kokkos::View<double**, Kokkos::LayoutRight>& disp_y_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& disp_z_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3_, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_vor_, Kokkos::View<double**, Kokkos::LayoutRight>& source_vals_div_,
                                Kokkos::View<double**, Kokkos::LayoutRight>& interp_vals_, Kokkos::View<CubedSpherePanel*> cubed_sphere_panels_, 
                                Kokkos::View<interact_pair*>& interaction_list_, int offset_, double kernel_eps_) : disp_x(disp_x_), disp_y(disp_y_), disp_z(disp_z_), target_pots_1(target_pots_1_), target_pots_2(target_pots_2_), 
                                target_pots_3(target_pots_3_), source_vals_vor(source_vals_vor_), source_vals_div(source_vals_div_), interp_vals(interp_vals_), cubed_sphere_panels(cubed_sphere_panels_), 
                                interaction_list(interaction_list_), offset(offset_), kernel_eps(kernel_eps_) {}

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

        double xi_s, eta_s, xi_t, eta_t, xyz_t[3], xyz_s[3], dp, gfcomp, x_s, y_s, z_s, gfcomp1, gfcomp2, dp2;
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
                dp2 = xyz_t[0]*xyz_s[0] + xyz_t[1]*xyz_s[1] + xyz_t[2]*xyz_s[2];
                // if (dp < 1.0 - 1e-15) { // constant is -1/(4pi)
                gfcomp = 0.07957747154594767 / (1.0 - dp + kernel_eps);
                gfcomp1 = -gfcomp * source_vals_vor(source_panel, k);
                gfcomp2 = 0.07957747154594767 / (1.0 - dp2 + kernel_eps) * source_vals_div(source_panel, k);
                Kokkos::atomic_add(&target_pots_1(target_panel, j), gfcomp1*(xyz_t[1]*z_s - xyz_t[2]*y_s));
                Kokkos::atomic_add(&target_pots_2(target_panel, j), gfcomp1*(xyz_t[2]*x_s - xyz_t[0]*z_s));
                Kokkos::atomic_add(&target_pots_3(target_panel, j), gfcomp1*(xyz_t[0]*y_s - xyz_t[1]*x_s));
                Kokkos::atomic_add(&target_pots_1(target_panel, j), gfcomp2*(xyz_s[0] - xyz_t[0] * dp2));
                Kokkos::atomic_add(&target_pots_2(target_panel, j), gfcomp2*(xyz_s[1] - xyz_t[1] * dp2));
                Kokkos::atomic_add(&target_pots_3(target_panel, j), gfcomp2*(xyz_s[2] - xyz_t[2] * dp2));
                // } 
            }
        }
    }
};

struct swe_vel_panel_interaction_ll {
    Kokkos::View<double**, Kokkos::LayoutRight> xcos;
    Kokkos::View<double**, Kokkos::LayoutRight> ycos;
    Kokkos::View<double**, Kokkos::LayoutRight> zcos;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_1;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_2;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_3;
    Kokkos::View<double**, Kokkos::LayoutRight> vel_x;
    Kokkos::View<double**, Kokkos::LayoutRight> vel_y;
    Kokkos::View<double**, Kokkos::LayoutRight> vel_z;
    Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_vors;
    Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_divs;
    Kokkos::View<double**, Kokkos::LayoutRight> vors;
    Kokkos::View<double**, Kokkos::LayoutRight> divs;
    Kokkos::View<double**, Kokkos::LayoutRight> area;
    Kokkos::View<int**, Kokkos::LayoutRight> leaf_panel_points;
    Kokkos::View<CubedSpherePanel*> cubed_sphere_panels;
    Kokkos::View<interact_pair*> interaction_list;
    double kernel_eps;
    int degree;

    swe_vel_panel_interaction_ll(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_vors_, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_divs_,
                                Kokkos::View<double**, Kokkos::LayoutRight>& vors_, Kokkos::View<double**, Kokkos::LayoutRight>& divs_, Kokkos::View<double**, Kokkos::LayoutRight>& area_, Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points_, 
                                Kokkos::View<CubedSpherePanel*> cubed_sphere_panels_, Kokkos::View<interact_pair*>& interaction_list_, double kernel_eps_, int degree_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), 
                                target_pots_1(target_pots_1_), target_pots_2(target_pots_2_), target_pots_3(target_pots_3_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), proxy_source_vors(proxy_source_vors_), proxy_source_divs(proxy_source_divs_), vors(vors_), divs(divs_), area(area_), leaf_panel_points(leaf_panel_points_), 
                                cubed_sphere_panels(cubed_sphere_panels_), interaction_list(interaction_list_), kernel_eps(kernel_eps_), degree(degree_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        int target_panel = interaction_list(i).target_panel;
        int source_panel = interaction_list(i).source_panel;
        int target_count, source_count, index, index_i, index_j, lon_count;
        int i_t = interaction_list(i).interact_type;
        lon_count = xcos.extent_int(1);
        double s_x[484], s_y[484], s_z[484], t_x[484], t_y[484], t_z[484], xyz[3], sv[484], sd[484];
        double xi, eta, min_xi, max_xi, min_eta, max_eta, xi_scale, xi_off, eta_scale, eta_off;
        if ((i_t== 0) or (i_t == 1)) {
            target_count = cubed_sphere_panels(target_panel).point_count;
            for (int j = 0; j < target_count; j++) {
                index = leaf_panel_points(target_panel,j);
                index_i = index / lon_count;
                index_j = index % lon_count;
                t_x[j] = xcos(index_i,index_j);
                t_y[j] = ycos(index_i,index_j);
                t_z[j] = zcos(index_i,index_j);
            }
        } else {
            target_count = (degree+1)*(degree+1);
            min_xi = cubed_sphere_panels(target_panel).min_xi;
            max_xi = cubed_sphere_panels(target_panel).max_xi;
            min_eta = cubed_sphere_panels(target_panel).min_eta;
            max_eta = cubed_sphere_panels(target_panel).max_eta;
            xi_off = 0.5*(min_xi+max_xi);
            xi_scale = 0.5*(max_xi-min_xi);
            eta_off = 0.5*(min_eta+max_eta);
            eta_scale = 0.5*(max_eta-min_eta);
            for (int j = 0; j < degree+1; j++) { // xi loop
                for (int k = 0; k < degree+1; k++) { // eta loop
                    xi = Kokkos::cos(Kokkos::numbers::pi*j/degree)*xi_scale+xi_off;
                    eta = Kokkos::cos(Kokkos::numbers::pi*k/degree)*eta_scale+eta_off;
                    xyz_from_xieta(xi, eta, cubed_sphere_panels(target_panel).face, xyz);
                    index = j*(degree+1)+k;
                    t_x[index]=xyz[0];
                    t_y[index]=xyz[1];
                    t_z[index]=xyz[2];
                }
            }
        }
        if ((i_t == 0) or (i_t == 2)) {
            source_count = cubed_sphere_panels(source_panel).point_count;
            for (int j = 0; j < source_count; j++) {
                index = leaf_panel_points(source_panel, j);
                index_i = index / lon_count;
                index_j = index % lon_count;
                s_x[j] = xcos(index_i,index_j);
                s_y[j] = ycos(index_i,index_j);
                s_z[j] = zcos(index_i,index_j);
                sv[j] = vors(index_i,index_j) * area(index_i,index_j);
                sd[j] = divs(index_i,index_j) * area(index_i,index_j);
            }
        } else {
            source_count = (degree+1)*(degree+1);
            min_xi = cubed_sphere_panels(source_panel).min_xi;
            max_xi = cubed_sphere_panels(source_panel).max_xi;
            min_eta = cubed_sphere_panels(source_panel).min_eta;
            max_eta = cubed_sphere_panels(source_panel).max_eta;
            xi_off = 0.5*(min_xi+max_xi);
            xi_scale = 0.5*(max_xi-min_xi);
            eta_off = 0.5*(min_eta+max_eta);
            eta_scale = 0.5*(max_eta-min_eta);
            for (int j = 0; j < degree+1; j++) { // xi loop
                for (int k = 0; k < degree+1; k++) { // eta loop
                    xi = Kokkos::cos(Kokkos::numbers::pi*j/degree)*xi_scale+xi_off;
                    eta = Kokkos::cos(Kokkos::numbers::pi*k/degree)*eta_scale+eta_off;
                    xyz_from_xieta(xi, eta, cubed_sphere_panels(source_panel).face, xyz);
                    index = j*(degree+1)+k;
                    s_x[index]=xyz[0];
                    s_y[index]=xyz[1];
                    s_z[index]=xyz[2];
                    sv[index] = proxy_source_vors(source_panel,index);
                    sd[index] = proxy_source_divs(source_panel,index);
                }
            }
        }
        double x_t, y_t, z_t, x_s, y_s, z_s, dp, gf, gf1, gf2, vx, vy, vz;
        for (int j = 0; j < target_count; j++) {
            x_t = t_x[j];
            y_t = t_y[j];
            z_t = t_z[j];
            index = leaf_panel_points(target_panel,j);
            index_i = index / lon_count;
            index_j = index % lon_count;
            for (int k = 0; k < source_count; k++) {
                x_s = s_x[k];
                y_s = s_y[k];
                z_s = s_z[k];
                dp = x_t*x_s + y_t*y_s + z_t*z_s;
                // constant is -1/(4pi)
                gf = 0.07957747154594767 / (1.0 - dp + kernel_eps);
                gf1 = -gf*sv[k];
                gf2 = gf*sd[k];
                vx = gf1*(y_t*z_s - z_t*y_s)+gf2*(x_s - x_t * dp);
                vy = gf1*(z_t*x_s - x_t*z_s)+gf2*(y_s - y_t * dp);
                vz = gf1*(x_t*y_s - y_t*x_s)+gf2*(z_s - z_t * dp);
                if ((i_t== 0) or (i_t == 1)) {
                    Kokkos::atomic_add(&vel_x(index_i, index_j), vx);
                    Kokkos::atomic_add(&vel_y(index_i, index_j), vy);
                    Kokkos::atomic_add(&vel_z(index_i, index_j), vz);
                } else {
                    Kokkos::atomic_add(&target_pots_1(target_panel,j), vx);
                    Kokkos::atomic_add(&target_pots_2(target_panel,j), vy);
                    Kokkos::atomic_add(&target_pots_3(target_panel,j), vz);
                }
            }
        }
    }
};

struct swe_vel_panel_interaction_octo {
    Kokkos::View<double**, Kokkos::LayoutRight> xcos;
    Kokkos::View<double**, Kokkos::LayoutRight> ycos;
    Kokkos::View<double**, Kokkos::LayoutRight> zcos;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_1;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_2;
    Kokkos::View<double**, Kokkos::LayoutRight> target_pots_3;
    Kokkos::View<double**, Kokkos::LayoutRight> vel_x;
    Kokkos::View<double**, Kokkos::LayoutRight> vel_y;
    Kokkos::View<double**, Kokkos::LayoutRight> vel_z;
    Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_vors;
    Kokkos::View<double**, Kokkos::LayoutRight> proxy_source_divs;
    Kokkos::View<double**, Kokkos::LayoutRight> vors;
    Kokkos::View<double**, Kokkos::LayoutRight> divs;
    Kokkos::View<double**, Kokkos::LayoutRight> area;
    Kokkos::View<int**, Kokkos::LayoutRight> leaf_panel_points;
    Kokkos::View<OctoSpherePanel*> octo_sphere_panels;
    Kokkos::View<interact_pair*> interaction_list;
    double kernel_eps;
    int degree;

    swe_vel_panel_interaction_octo(Kokkos::View<double**, Kokkos::LayoutRight>& xcos_, Kokkos::View<double**, Kokkos::LayoutRight>& ycos_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& zcos_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_1_, Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_2_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& target_pots_3_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_x_, Kokkos::View<double**, Kokkos::LayoutRight>& vel_y_, 
                                Kokkos::View<double**, Kokkos::LayoutRight>& vel_z_, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_vors_, Kokkos::View<double**, Kokkos::LayoutRight>& proxy_source_divs_,
                                Kokkos::View<double**, Kokkos::LayoutRight>& vors_, Kokkos::View<double**, Kokkos::LayoutRight>& divs_, Kokkos::View<double**, Kokkos::LayoutRight>& area_, Kokkos::View<int**, Kokkos::LayoutRight>& leaf_panel_points_, 
                                Kokkos::View<OctoSpherePanel*> octo_sphere_panels_, Kokkos::View<interact_pair*>& interaction_list_, double kernel_eps_, int degree_) : xcos(xcos_), ycos(ycos_), zcos(zcos_), 
                                target_pots_1(target_pots_1_), target_pots_2(target_pots_2_), target_pots_3(target_pots_3_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), proxy_source_vors(proxy_source_vors_), proxy_source_divs(proxy_source_divs_), vors(vors_), divs(divs_), area(area_), leaf_panel_points(leaf_panel_points_), 
                                octo_sphere_panels(octo_sphere_panels_), interaction_list(interaction_list_), kernel_eps(kernel_eps_), degree(degree_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        int target_panel = interaction_list(i).target_panel;
        int source_panel = interaction_list(i).source_panel;
        int target_count, source_count, index, index_i, index_j, lon_count;
        int i_t = interaction_list(i).interact_type;
        lon_count = xcos.extent_int(1);
        // double s_x[484], s_y[484], s_z[484], t_x[484], t_y[484], t_z[484], xyz[3], sv[484], sd[484];
        // std::cout << i << std::endl;
        double s_x[2025], s_y[2025], s_z[2025], t_x[2025], t_y[2025], t_z[2025], xyz[3], sv[2025], sd[2025];
        // std::cout << i << std::endl;
        double lat, lon, min_lat, max_lat, min_lon, max_lon, lat_scale, lat_off, lon_scale, lon_off;
        // std::cout << i << " " << i_t << std::endl;
        if ((i_t== 0) or (i_t == 1)) {
            target_count = octo_sphere_panels(target_panel).point_count;
            for (int j = 0; j < target_count; j++) {
                index = leaf_panel_points(target_panel,j);
                index_i = index / lon_count;
                index_j = index % lon_count;
                t_x[j] = xcos(index_i,index_j);
                t_y[j] = ycos(index_i,index_j);
                t_z[j] = zcos(index_i,index_j);
            }
        } else {
            target_count = (degree+1)*(degree+1);
            min_lat = octo_sphere_panels(target_panel).min_lat;
            max_lat = octo_sphere_panels(target_panel).max_lat;
            min_lon = octo_sphere_panels(target_panel).min_lon;
            max_lon = octo_sphere_panels(target_panel).max_lon;
            lat_off = 0.5*(min_lat+max_lat);
            lat_scale = 0.5*(max_lat-min_lat);
            lon_off = 0.5*(min_lon+max_lon);
            lon_scale = 0.5*(max_lon-min_lon);
            for (int j = 0; j < degree+1; j++) { // xi loop
                for (int k = 0; k < degree+1; k++) { // eta loop
                    lat = Kokkos::cos(Kokkos::numbers::pi*j/degree)*lat_scale+lat_off;
                    lon = Kokkos::cos(Kokkos::numbers::pi*k/degree)*lon_scale+lon_off;
                    xyz_from_lonlat(xyz[0], xyz[1], xyz[2], lon*Kokkos::numbers::pi/180.0, lat*Kokkos::numbers::pi/180.0);
                    index = j*(degree+1)+k;
                    t_x[index]=xyz[0];
                    t_y[index]=xyz[1];
                    t_z[index]=xyz[2];
                }
            }
        }
        if ((i_t == 0) or (i_t == 2)) {
            source_count = octo_sphere_panels(source_panel).point_count;
            for (int j = 0; j < source_count; j++) {
                index = leaf_panel_points(source_panel, j);
                index_i = index / lon_count;
                index_j = index % lon_count;
                s_x[j] = xcos(index_i,index_j);
                s_y[j] = ycos(index_i,index_j);
                s_z[j] = zcos(index_i,index_j);
                sv[j] = vors(index_i,index_j) * area(index_i,index_j);
                sd[j] = divs(index_i,index_j) * area(index_i,index_j);
            }
        } else {
            source_count = (degree+1)*(degree+1);
            min_lat = octo_sphere_panels(source_panel).min_lat;
            max_lat = octo_sphere_panels(source_panel).max_lat;
            min_lon = octo_sphere_panels(source_panel).min_lon;
            max_lon = octo_sphere_panels(source_panel).max_lon;
            lat_off = 0.5*(min_lat+max_lat);
            lat_scale = 0.5*(max_lat-min_lat);
            lon_off = 0.5*(min_lon+max_lon);
            lon_scale = 0.5*(max_lon-min_lon);
            for (int j = 0; j < degree+1; j++) { // xi loop
                for (int k = 0; k < degree+1; k++) { // eta loop
                    lat = Kokkos::cos(Kokkos::numbers::pi*j/degree)*lat_scale+lat_off;
                    lon = Kokkos::cos(Kokkos::numbers::pi*k/degree)*lon_scale+lon_off;
                    xyz_from_lonlat(xyz[0], xyz[1], xyz[2], lon*Kokkos::numbers::pi/180.0, lat*Kokkos::numbers::pi/180.0);
                    index = j*(degree+1)+k;
                    s_x[index]=xyz[0];
                    s_y[index]=xyz[1];
                    s_z[index]=xyz[2];
                    sv[index] = proxy_source_vors(source_panel,index);
                    sd[index] = proxy_source_divs(source_panel,index);
                }
            }
        }
        double x_t, y_t, z_t, x_s, y_s, z_s, dp, gf, gf1, gf2, vx, vy, vz;
        for (int j = 0; j < target_count; j++) {
            x_t = t_x[j];
            y_t = t_y[j];
            z_t = t_z[j];
            
            for (int k = 0; k < source_count; k++) {
                x_s = s_x[k];
                y_s = s_y[k];
                z_s = s_z[k];
                dp = x_t*x_s + y_t*y_s + z_t*z_s;
                // constant is -1/(4pi)
                gf = 0.07957747154594767 / (1.0 - dp + kernel_eps);
                gf1 = -gf*sv[k];
                gf2 = -gf*sd[k];
                vx = gf1*(y_t*z_s - z_t*y_s)+gf2*(x_s - x_t * dp);
                vy = gf1*(z_t*x_s - x_t*z_s)+gf2*(y_s - y_t * dp);
                vz = gf1*(x_t*y_s - y_t*x_s)+gf2*(z_s - z_t * dp);
                if ((i_t== 0) or (i_t == 1)) {
                    index = leaf_panel_points(target_panel,j);
                    index_i = index / lon_count;
                    index_j = index % lon_count;
                    Kokkos::atomic_add(&vel_x(index_i, index_j), vx);
                    Kokkos::atomic_add(&vel_y(index_i, index_j), vy);
                    Kokkos::atomic_add(&vel_z(index_i, index_j), vz);
                } else {
                    Kokkos::atomic_add(&target_pots_1(target_panel,j), vx);
                    Kokkos::atomic_add(&target_pots_2(target_panel,j), vy);
                    Kokkos::atomic_add(&target_pots_3(target_panel,j), vz);
                }
            }
        }
    }
};

#endif
