add_library(specBSLPM
    cubed_sphere_transforms.cpp
    direct_sum.cpp
    general_utils.cpp
    initial_conditions.cpp
    initialize_cubed_sphere.cpp
    io_utils.cpp
    mpi_utils.cpp)

target_link_libraries(specBSLPM PUBLIC MPI::MPI_C LAPACK::LAPACK Kokkos::kokkos)
target_include_directories(specBSLPM PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>)

set_property(TARGET specBSLPM PROPERTY POSITION_INDEPENDENT_CODE ON)

install(FILES
    # all FastSphereSums header/include files go here
    ${CMAKE_BINARY_DIR}/specBSLPM-config.h
    cubed_sphere_transforms.hpp
    cubed_sphere_transforms_impl.hpp
    direct_sum.hpp
    general_utils.hpp
    general_utils_impl.hpp
    initial_conditions.hpp
    initialize_cubed_sphere.hpp
    io_utils.hpp
    mpi_utils.hpp
    run_config.hpp
    DESTINATION ${PROJECT_SOURCE_DIR}/include)