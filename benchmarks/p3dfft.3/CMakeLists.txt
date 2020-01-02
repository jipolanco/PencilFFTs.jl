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
        COMMAND julia --project
            -e "import FFTW_jll; println(FFTW_jll.LIBPATH)"
        OUTPUT_VARIABLE julia_fftw3_libpath
        RESULT_VARIABLE julia_result
        )
    if(NOT julia_result EQUAL 0)
        message(FATAL_ERROR "julia process failed!")
    endif()

    string(STRIP ${julia_fftw3_libpath} fftw3_libdir)
    get_filename_component(FFTW3_HOME "${fftw3_libdir}/.." ABSOLUTE)

    find_library(FFTW3_LIB fftw3
        HINTS ${fftw3_libdir} NO_DEFAULT_PATH)
    find_file(FFTW3_INC fftw3.h
        HINTS "${FFTW3_HOME}/include" NO_DEFAULT_PATH)

    message(STATUS "${FFTW3_LIB} ${FFTW3_INC}")

    if(NOT FFTW3_INC OR NOT FFTW3_LIB)
        message(SEND_ERROR "fftw3.h or libfftw3.so not found!")
    endif()

    get_filename_component(fftw3_incdir ${FFTW3_INC} DIRECTORY)
else()
    set(FFTW3_LIBDIR "" CACHE PATH "FFTW3 library directory")
    set(FFTW3_INCDIR "" CACHE PATH "FFTW3 include directory")

    find_library(FFTW3_LIB NAMES fftw3 HINTS ${FFTW3_LIBDIR})
    find_file(FFTW3_INC NAMES fftw3.h HINTS ${FFTW3_INCDIR})

    if(NOT FFTW3_INC OR NOT FFTW3_LIB)
        message(FATAL_ERROR
            "FFTW3 not found! Try setting FFTW3_LIBDIR and FFTW3_INCDIR.")
    endif()

    get_filename_component(fftw3_incdir "${FFTW3_INC}" DIRECTORY)
    get_filename_component(fftw3_libdir "${FFTW3_LIB}" DIRECTORY)

    message(STATUS "${fftw3_incdir} ${fftw3_libdir}")

    # P3DFFT expects a FFTW3 directory structure of the form
    # FFTW3_HOME/{lib,include}. This is not compatible with the way FFTW3 is
    # commonly installed in Linux distros. We workaround this by creating
    # symlinks.
    set(FFTW3_HOME "${CMAKE_CURRENT_BINARY_DIR}/fftw3")
    file(MAKE_DIRECTORY "${FFTW3_HOME}")
    file(CREATE_LINK "${fftw3_libdir}" "${FFTW3_HOME}/lib" SYMBOLIC)
    file(CREATE_LINK "${fftw3_incdir}" "${FFTW3_HOME}/include" SYMBOLIC)
endif()

find_library(FFTW3_LIB_F NAMES libfftw3f.so HINTS ${fftw3_libdir})

message(STATUS "FFTW3 home: ${FFTW3_HOME}")
message(STATUS "Using FFTW3 library dir: ${fftw3_libdir}")
message(STATUS "Using FFTW3 include dir: ${fftw3_incdir}")

# Install P3DFFT++ as external project.
set(p3dfft_git_commit_hash 09040fd71b6d6e90e18808b1a0dfa6b41c5dbc8b)
set(p3dfft_install_prefix ${CMAKE_CURRENT_BINARY_DIR}/p3dfft)

set(P3DFFT_CONFIG_FLAGS "--enable-fftwestimate"
    CACHE STRING "P3DFFT configuration flags")
set(P3DFFT_CXX_FLAGS "-O3 -DNDEBUG"
    CACHE STRING "P3DFFT build flags")

set(p3dfft_config_flags
    "${P3DFFT_CONFIG_FLAGS}"
    "--enable-fftw"
    "--with-fftw=${FFTW3_HOME}"
    "--prefix=${p3dfft_install_prefix}"
    )
set(p3dfft_config_vars
    "CXX=${MPI_CXX_COMPILER}"
    "CXXFLAGS=${P3DFFT_CXX_FLAGS}")
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
    INTERFACE_INCLUDE_DIRECTORIES "${p3dfft_incdir};${fftw3_incdir}"
    INTERFACE_COMPILE_DEFINITIONS "${p3dfft_interface_definitions}"
    )
target_link_libraries(p3dfft INTERFACE ${FFTW3_LIB} ${FFTW3_LIB_F})

add_executable(bench_p3dfft bench_p3dfft.cpp)
target_link_libraries(bench_p3dfft p3dfft ${MPI_CXX_LIBRARIES})
