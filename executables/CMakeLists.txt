add_executable(test test.cpp)
target_link_libraries(test PUBLIC specBSLPM)
target_include_directories(test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)

add_executable(poisson_eq_ds poisson_eq_ds.cpp)
target_link_libraries(poisson_eq_ds PUBLIC specBSLPM)
target_include_directories(poisson_eq_ds PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)