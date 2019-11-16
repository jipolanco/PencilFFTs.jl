cmake_minimum_required(VERSION 3.5)
project(bench_p3dfft++ CXX)

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
    find_package(MPI REQUIRED COMPONENTS CXX)
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
        COMMAND julia -e "import FFTW; println(FFTW.libfftw3)"
        OUTPUT_VARIABLE julia_libfftw3
        RESULT_VARIABLE julia_result
        )
    if(NOT julia_result EQUAL 0)
        message(FATAL_ERROR "julia process failed!")
    endif()

    string(STRIP ${julia_libfftw3} FFTW3_LIB)
    get_filename_component(FFTW3_LIBDIR ${FFTW3_LIB} DIRECTORY CACHE)
    find_file(FFTW3_INC fftw3.h HINTS ${FFTW3_LIBDIR}/../include)

    if(NOT FFTW3_INC)
        message(SEND_ERROR "fftw3.h included in FFTW.jl not found!")
    endif()

    get_filename_component(FFTW3_INCDIR ${FFTW3_INC} DIRECTORY CACHE)
else()
    set(FFTW3_LIBDIR "" CACHE PATH "FFTW3 library directory")
    set(FFTW3_INCDIR "" CACHE PATH "FFTW3 include directory")

    find_library(FFTW3_LIB NAMES libfftw3.so HINTS ${FFTW3_LIBDIR})
    find_file(FFTW3_INC NAMES fftw3.h HINTS ${FFTW3_INCDIR})

    if(NOT FFTW3_INC OR NOT FFTW3_LIB)
        message(SEND_ERROR
            "FFTW3 not found! Try setting FFTW3_LIBDIR and FFTW3_INCDIR.")
    endif()
endif()

find_library(FFTW3_LIB_F NAMES libfftw3f.so HINTS ${FFTW3_LIBDIR})

message(STATUS "Using FFTW3 library dir: ${FFTW3_LIBDIR}")
message(STATUS "Using FFTW3 include dir: ${FFTW3_INCDIR}")

# Install P3DFFT++ as external project.
set(p3dfft_git_commit_hash 1ac34efd1ac6fcb989e26e89d2661f2f7306dd03)
set(p3dfft_install_prefix ${CMAKE_CURRENT_BINARY_DIR}/p3dfft)
set(p3dfft_config_flags
    "--enable-fftw"
    "--enable-fftwestimate"
    "--with-fftw-lib=${FFTW3_LIBDIR}"
    "--with-fftw-inc=${FFTW3_INCDIR}"
    "--prefix=${p3dfft_install_prefix}"
    )
set(p3dfft_config_vars
    "CXX=${MPI_CXX_COMPILER}"
    "CFLAGS=-O3 -DNDEBUG")
set(p3dfft_interface_definitions "HAVE_CONFIG_H")

option(P3DFFT_ENABLE_TIMERS "Enable P3DFFT timers" ON)
if(P3DFFT_ENABLE_TIMERS)
    list(APPEND p3dfft_config_flags "--enable-timers")
    list(APPEND p3dfft_interface_definitions "TIMERS")
endif()

message(STATUS "P3DFFT++ config flags: ${p3dfft_config_flags}")
message(STATUS "P3DFFT++ config variables: ${p3dfft_config_vars}")

find_program(MAKE_EXE NAMES make)

set(p3dfft_incdir ${p3dfft_install_prefix}/include)
file(MAKE_DIRECTORY ${p3dfft_incdir})

ExternalProject_Add(p3dfft_build
    PREFIX p3dfft.build
    GIT_REPOSITORY https://github.com/sdsc/p3dfft.3
    GIT_TAG ${p3dfft_git_commit_hash}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE TRUE
    CONFIGURE_COMMAND ./configure ${p3dfft_config_flags} ${p3dfft_config_vars}
    BUILD_COMMAND ${MAKE_EXE} -C build
    INSTALL_COMMAND ${MAKE_EXE} -C build install
    COMMAND ${MAKE_EXE} install-data-am  # install header files
    )

# Add interface to P3DFFT
add_library(p3dfft STATIC IMPORTED GLOBAL)
add_dependencies(p3dfft p3dfft_build)
set_target_properties(p3dfft PROPERTIES
    IMPORTED_LOCATION ${p3dfft_install_prefix}/lib/libp3dfft.3.a
    INTERFACE_INCLUDE_DIRECTORIES "${p3dfft_incdir};${FFTW3_INCDIR}"
    INTERFACE_COMPILE_DEFINITIONS ${p3dfft_interface_definitions}
    )
target_link_libraries(p3dfft INTERFACE ${FFTW3_LIB} ${FFTW3_LIB_F})

add_executable(bench_p3dfft bench_p3dfft.cpp)
target_link_libraries(bench_p3dfft p3dfft ${MPI_CXX_LIBRARIES})
