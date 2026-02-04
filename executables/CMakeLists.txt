add_executable(test test.cpp)
target_link_libraries(test PUBLIC specBSLPM MPI::MPI_C Kokkos::kokkos KokkosKernels::kokkoskernels netCDF::netcdf)
target_include_directories(test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)

add_executable(poisson_eq_ds poisson_eq_ds.cpp)
target_link_libraries(poisson_eq_ds PUBLIC specBSLPM MPI::MPI_C Kokkos::kokkos KokkosKernels::kokkoskernels netCDF::netcdf)
target_include_directories(poisson_eq_ds PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)

add_executable(poisson_eq_fmm poisson_eq_fmm.cpp)
target_link_libraries(poisson_eq_fmm PUBLIC specBSLPM MPI::MPI_C Kokkos::kokkos KokkosKernels::kokkoskernels netCDF::netcdf)
target_include_directories(poisson_eq_fmm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)

add_executable(deriv_test deriv_test.cpp)
target_link_libraries(deriv_test PUBLIC specBSLPM MPI::MPI_C Kokkos::kokkos KokkosKernels::kokkoskernels netCDF::netcdf)
target_include_directories(deriv_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)

add_executable(bve_time_evo bve_time_evo.cpp)
target_link_libraries(bve_time_evo PUBLIC specBSLPM MPI::MPI_C Kokkos::kokkos KokkosKernels::kokkoskernels netCDF::netcdf)
target_include_directories(bve_time_evo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)

add_executable(swe_time_evo swe_time_evo.cpp)
target_link_libraries(swe_time_evo PUBLIC specBSLPM MPI::MPI_C Kokkos::kokkos KokkosKernels::kokkoskernels netCDF::netcdf)
target_include_directories(swe_time_evo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)