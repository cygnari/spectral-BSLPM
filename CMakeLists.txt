function (printvar var)
    message("${var}: ${${var}}")
endfunction()

cmake_minimum_required(VERSION 3.10)
project(specBSLPM VERSION 1.0 LANGUAGES CXX C)
set(default_build_type Release)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
option(BUILD_SHARED_LIBS "Build shared libraries, rather than static" ON)

if (BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(CMAKE_CXX_FLAGS_RELEASE -O3)

# turn off deprecated env variable handling:
# cmake will now use $<Package>_ROOT env variables in its search for Package
# dependencies
CMAKE_POLICY(SET CMP0074 NEW)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.24)
  # turn off deprecated cmake download timestamp handling
  # cmake will now set the local timestamp of a dependency to match its package
  # creation timestamp, rather than its download timestamp.
  CMAKE_POLICY(SET CMP0135 NEW)
endif()

find_package(MPI REQUIRED COMPONENTS C)
find_package(Kokkos REQUIRED CONFIG)
find_package(LAPACK REQUIRED)

add_subdirectory(src)
add_subdirectory(executables)

# CMake auto-generated config settings will go in this file
configure_file(specBSLPM-config.h.in specBSLPM-config.h)