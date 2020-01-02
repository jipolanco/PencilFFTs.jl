cmake_minimum_required(VERSION 3.5)
project(bench_p3dfft CXX Fortran C)

include(ExternalProject)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compatibility: define "add_compile_definitions" in CMake < 3.12.
if(NOT COMMAND add_compile_definitions)
    function(add_compile_definitions)
        foreach(def IN LISTS ARGN)
            add_definitions("-D${def}")
        endforeach()
    endfunction()
endif()

# Find MPI
if(NOT MPI_CXX_FOUND)
    find_package(MPI REQUIRED COMPONENTS CXX C Fortran)
    set(MPI_CXX_SKIP_MPICXX ON CACHE INTERNAL "" FORCE)  # skip MPI C++ API
    message(STATUS "MPI compiler:  ${MPI_CXX_COMPILER}")
    message(STATUS "MPI libraries: ${MPI_CXX_LIBRARIES}")
endif()

include_directories(${MPI_CXX_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
add_compile_options(${MPI_CXX_COMPILE_OPTIONS})
add_compile_definitions(${MPI_CXX_COMPILE_DEFINITIONS})

# Find FFTW.
option(USE_JULIA_FFTW "Use same FFTW library as Julia" ON)

if(USE_JULIA_FFTW)
    execute_process(
        COMMAND julia --project
            -e "import FFTW_jll; println(FFTW_jll.LIBPATH)"
        OUTPUT_VARIABLE julia_fftw3_libpath
        RESULT_VARIABLE julia_result
        )
    if(NOT julia_result EQUAL 0)
        message(FATAL_ERROR "julia process failed!")
    endif()
    string(STRIP ${julia_fftw3_libpath} fftw3_libdir)
else()
    set(FFTW3_LIBDIR "" CACHE PATH "FFTW3 library directory")
    find_library(FFTW3_LIB NAMES fftw3 HINTS ${FFTW3_LIBDIR})
    if(NOT FFTW3_LIB)
        message(FATAL_ERROR
            "FFTW3 not found! Try setting FFTW3_LIBDIR.")
    endif()
    get_filename_component(fftw3_libdir "${FFTW3_LIB}" DIRECTORY)
endif()

message(STATUS "Using FFTW3 library dir: ${fftw3_libdir}")

find_library(fftw3_lib_actual NAMES fftw3 HINTS ${fftw3_libdir})
set(fftw3_home "${CMAKE_CURRENT_BINARY_DIR}/fftw3")
file(REMOVE_RECURSE ${fftw3_home})
file(MAKE_DIRECTORY "${fftw3_home}/lib")
set(fftw3_lib_symlink ${fftw3_home}/lib/libfftw3.so)
file(CREATE_LINK ${fftw3_lib_actual} ${fftw3_lib_symlink} SYMBOLIC)

# Install P3DFFT as external project.
set(p3dfft_git_repo https://github.com/sdsc/p3dfft.git)
set(p3dfft_git_commit_hash 673cf5ef6fc63cc83a5c9235627fd7e7b4d435a2)
set(p3dfft_install_prefix ${CMAKE_CURRENT_BINARY_DIR}/p3dfft)

set(P3DFFT_CONFIG_FLAGS "--enable-estimate;--enable-stride1"
    CACHE STRING "P3DFFT configuration flags")
set(P3DFFT_Fortran_FLAGS "-O3 -DNDEBUG" CACHE STRING "P3DFFT build flags")
set(P3DFFT_C_FLAGS "-O3 -DNDEBUG" CACHE STRING "P3DFFT build flags")

set(p3dfft_config_flags
    "${P3DFFT_CONFIG_FLAGS}"
    "--enable-fftw"
    "--with-fftw=${fftw3_home}"
    "--prefix=${p3dfft_install_prefix}"
)

message(STATUS "Fortran compiler ID: ${CMAKE_Fortran_COMPILER_ID}")
message(STATUS "C++ compiler ID: ${CMAKE_CXX_COMPILER_ID}")

if(CMAKE_Fortran_COMPILER_ID STREQUAL GNU)
    set(p3dfft_config_flags "${p3dfft_config_flags};--enable-gnu")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
    set(p3dfft_config_flags "${p3dfft_config_flags};--enable-intel")
endif()

set(p3dfft_config_vars
    "CC=${MPI_C_COMPILER}"
    "FC=${MPI_Fortran_COMPILER}"
    "CFLAGS=${P3DFFT_C_FLAGS}"
    "FCFLAGS=${P3DFFT_Fortran_FLAGS}"
    "LDFLAGS=-lfftw3"
)
set(p3dfft_interface_definitions "HAVE_CONFIG_H")

message(STATUS "P3DFFT config flags: ${p3dfft_config_flags}")
message(STATUS "P3DFFT config variables: ${p3dfft_config_vars}")

find_program(MAKE_EXE NAMES make)

set(p3dfft_incdir ${p3dfft_install_prefix}/include)
file(MAKE_DIRECTORY ${p3dfft_incdir})

ExternalProject_Add(p3dfft_build
    PREFIX p3dfft.build
    GIT_REPOSITORY ${p3dfft_git_repo}
    GIT_TAG ${p3dfft_git_commit_hash}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    # UPDATE_COMMAND ""

    # By default, P3DFFT looks for lib/libfftw3.a, but FFTW3_jll only includes
    # the shared library version (libfftw3.so). This patch makes P3DFFT look
    # for "libfftw3.so".
    PATCH_COMMAND sed -i -e s/libfftw3\.a/libfftw3\.so/ configure

    BUILD_IN_SOURCE TRUE
    CONFIGURE_COMMAND aclocal
    COMMAND automake
    COMMAND ./configure ${p3dfft_config_flags} ${p3dfft_config_vars}
    BUILD_COMMAND ${MAKE_EXE} -C build
    # COMMAND ${MAKE_EXE}
    INSTALL_COMMAND ${MAKE_EXE} -C build install
    COMMAND ${MAKE_EXE} install-data-am  # install header files
)

# Add interface to P3DFFT
add_library(p3dfft STATIC IMPORTED GLOBAL)
add_dependencies(p3dfft p3dfft_build)
set_target_properties(p3dfft PROPERTIES
    IMPORTED_LOCATION ${p3dfft_install_prefix}/lib/libp3dfft.a
    INTERFACE_INCLUDE_DIRECTORIES "${p3dfft_incdir}"
    # INTERFACE_COMPILE_DEFINITIONS "${p3dfft_interface_definitions}"
    IMPORTED_LINK_INTERFACE_LANGUAGES Fortran
)
target_link_libraries(p3dfft
    INTERFACE ${fftw3_lib_actual} ${MPI_Fortran_LIBRARIES})

add_executable(bench_p3dfft bench_p3dfft.cpp)
target_link_libraries(bench_p3dfft p3dfft ${MPI_CXX_LIBRARIES})
