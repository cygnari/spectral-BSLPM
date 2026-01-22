add_library(specBSLPM
    fmm_interactions/inverse_poisson.cpp
    fmm_interactions/bve_vel.cpp
    bve_time_step.cpp
    cubed_sphere_transforms.cpp
    deriv_funcs.cpp
    direct_sum.cpp
    fmm_funcs.cpp
    forcing_funcs.cpp
    general_utils.cpp
    initial_conditions.cpp
    initialize_cubed_sphere.cpp
    interp_funcs.cpp
    io_utils.cpp
    mpi_utils.cpp)

target_link_libraries(specBSLPM PUBLIC MPI::MPI_C LAPACK::LAPACK Kokkos::kokkos KokkosKernels::kokkoskernels netCDF::netcdf)
# target_link_libraries(specBSLPM PUBLIC MPI::MPI_C LAPACK::LAPACK Kokkos::kokkos)
target_include_directories(specBSLPM PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>)

set_property(TARGET specBSLPM PROPERTY POSITION_INDEPENDENT_CODE ON)

install(FILES
    # all FastSphereSums header/include files go here
    ${CMAKE_BINARY_DIR}/specBSLPM-config.h
    fmm_interactions/inverse_poisson.hpp
    fmm_interactions/inverse_poisson_impl.hpp
    fmm_interactions/bve_vel.hpp
    fmm_interactions/bve_vel_impl.hpp
    bve_time_step.hpp
    cubed_sphere_transforms.hpp
    cubed_sphere_transforms_impl.hpp
    deriv_funcs.hpp
    deriv_funcs_impl.hpp
    direct_sum.hpp
    direct_sum_impl.hpp
    fmm_funcs.hpp
    fmm_funcs_impl.hpp
    forcing_funcs.hpp
    general_utils.hpp
    general_utils_impl.hpp
    initial_conditions.hpp
    initialize_cubed_sphere.hpp
    interp_funcs.hpp
    io_utils.hpp
    mpi_utils.hpp
    run_config.hpp
    DESTINATION ${PROJECT_SOURCE_DIR}/include)