cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(accrqa CXX)

option(BUILD_APPLICATIONS "Build example C/C++ application" OFF)
option(BUILD_TESTS "Build test program" OFF)
option(BUILD_SHARED_LIBS "Build the shared library" ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
# A fancy way of exporting only function that should be visible from outside
#include(GenerateExportHeader) 

# Read version from a file
file(READ "version.txt" FVERSION)

# Separate version into major, minor, patch
string(REGEX MATCHALL "[^\.]+" FVERSTRINGS ${FVERSION})
list(GET FVERSTRINGS 0 F_VER_MAJOR)
list(GET FVERSTRINGS 1 F_VER_MINOR)
list(GET FVERSTRINGS 2 F_VER_PATCH)

set(ACCRQA_VERSION_MAJOR ${F_VER_MAJOR})
set(ACCRQA_VERSION_MINOR ${F_VER_MINOR})
set(ACCRQA_VERSION_PATCH ${F_VER_PATCH})
set(ACCRQA_VERSION_STRING ${ACCRQA_VERSION_MAJOR}.${ACCRQA_VERSION_MINOR}.${ACCRQA_VERSION_PATCH})
add_compile_definitions(ACCRQA_VER_MAJOR=${ACCRQA_VERSION_MAJOR})
add_compile_definitions(ACCRQA_VER_MINOR=${ACCRQA_VERSION_MINOR})
add_compile_definitions(ACCRQA_VER_PATCH=${ACCRQA_VERSION_PATCH})


# Add local location of cmake modules.
# In our case it is the CUDA script.
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)

# Get string length of current source directory for log messages.
#string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_SIZE)
#add_definitions(-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE})

if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CUDAToolkit_ROOT})
    set(CUDAToolkit_ROOT "$ENV{CUDAToolkit_ROOT}")
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDA_HOME})
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")
endif()

# Dependencies
if (FIND_CUDA OR NOT DEFINED FIND_CUDA)
    find_package(CUDAToolkit 7.0)
endif()
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    #add_definitions(-DCUDA_FOUND)
    add_compile_definitions(CUDA_FOUND)
endif()

# ----- OpenMP Cross-Platform Handling (macOS + Linux) -----
if(APPLE)
    message(STATUS "Configuring OpenMP for macOS...")
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_HOST_SYSTEM_VERSION VERSION_LESS 23)
        message(STATUS "macOS version 13 and lower")
    else()
        # ----------- MacPorts detection -----------
        if(EXISTS "/opt/local/lib/libomp/libomp.dylib")
            message(STATUS "Found MacPorts OpenMP (libomp)")
         
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/local/include -I/opt/local/include/libomp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/opt/local/lib/libomp/libomp.dylib")
        
        # ----------- Homebrew (Intel & ARM) detection -----------    
        elseif(EXISTS "/opt/homebrew/opt/libomp/lib/libomp.dylib")
            message(STATUS "Found Homebrew OpenMP (Apple Silicon)")
        
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
        
        elseif(EXISTS "/usr/local/opt/libomp/lib/libomp.dylib")
            message(STATUS "Found Homebrew OpenMP (Intel Mac)")
        
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
        
        else()
            message(FATAL_ERROR "OpenMP not found! Please install libomp (MacPorts or Homebrew).")
        endif()
        
        # Apply flags (compiler + linker)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${OpenMP_omp_LIBRARY}/../ -lomp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${OpenMP_omp_LIBRARY}/../ -lomp")
    endif()
else()
    # ----- Linux: standard OpenMP -----
    find_package(OpenMP REQUIRED COMPONENTS C CXX)
endif()

# Set install directory paths.
if (PYTHON_EXECUTABLE)
    set(ACCRQA_LIB_INSTALL_DIR ${CMAKE_PROJECT_NAME})
    set(ACCRQA_INCLUDE_INSTALL_DIR ${CMAKE_PROJECT_NAME})
    set(ACCRQA_BIN_INSTALL_DIR ${CMAKE_PROJECT_NAME})
else()
    if (NOT ACCRQA_LIB_INSTALL_DIR)
        set(ACCRQA_LIB_INSTALL_DIR "lib")
    endif()
    if (NOT ACCRQA_INCLUDE_INSTALL_DIR)
        set(ACCRQA_INCLUDE_INSTALL_DIR "include")
    endif()
    if (NOT ACCRQA_BIN_INSTALL_DIR)
        set(ACCRQA_BIN_INSTALL_DIR "bin")
    endif()
endif()

# Include our cmake module with CUDA and compiler options
include(compiler_options)

# Generate source list from subdirectories.
add_subdirectory(src)

# Build example application.
if(BUILD_APPLICATIONS)
    add_subdirectory(application)
endif()

# Build tests.
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()


# Build library using list of source files.
if (CUDAToolkit_FOUND)
    # Add CUDA kernel sources.
    list(APPEND ${CMAKE_PROJECT_NAME}_SRC ${${CMAKE_PROJECT_NAME}_CUDA_SRC})
    add_library(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_SRC})
    target_link_libraries(${CMAKE_PROJECT_NAME} CUDA::cudart_static)
else()
    add_library(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_SRC})
endif()



# Not sure if this does anything. Commenting out.
if (PYTHON_EXECUTABLE)
    message(STATUS "Building for Python: ${PYTHON_EXECUTABLE}")
endif()

message("-- INFO: Using R library destination: ${CMAKE_R_LIB_DIR}")
message("-- INFO: Using R include destination: ${CMAKE_R_INC_DIR}")
if (CMAKE_R_LIB_DIR AND CMAKE_R_INC_DIR)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_R_INC_DIR})
    target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_R_LIB_DIR})
    add_definitions( -DACCRQA_R_FOUND )
else()
    message("-- INFO: R library not found. Not compiling R bindings.")
endif()

# --- OpenMP handling: use MacPorts libomp on macOS, otherwise use OpenMP::OpenMP_CXX ---
if(APPLE)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_HOST_SYSTEM_VERSION VERSION_LESS 23)
        message(STATUS "macOS version 13 and lower")
    else()
        message(STATUS "Using MacPorts libomp for OpenMP on macOS")
        
        # Compiler flags for AppleClang
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/local/include -I/opt/local/include/libomp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        
        # Linker flags pointing to MacPorts lib
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/local/lib/libomp -lomp")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/opt/local/lib/libomp -lomp")
        
        # Also try to provide the variables that some CMake scripts check
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/local/lib/libomp/libomp.dylib")
    endif()
else()
    # normal systems: rely on find_package(OpenMP)
    target_link_libraries(${CMAKE_PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# Since we are doing brute force using CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS it is
# not needed
#generate_export_header(${CMAKE_PROJECT_NAME}
#    EXPORT_FILE_NAME ${CMAKE_PROJECT_NAME}/${CMAKE_PROJECT_NAME}_export.h
#    )

#set(public_header_files
#    include/AccRQA_library.hpp
#    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}/${CMAKE_PROJECT_NAME}_export.h
#    )

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    LANGUAGES CXX
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    VERSION ${ACCRQA_VERSION_STRING}
    SOVERSION ${ACCRQA_VERSION_MAJOR}
#    PUBLIC_HEADER "${public_header_files}"
#    CXX_VISIBILITY_PRESET visible
#    VISIBILITY_INLINES_HIDDEN 0
    )
target_sources(${CMAKE_PROJECT_NAME}
    PRIVATE
#        ${public_header_files}
        ${${CMAKE_PROJECT_NAME}_SRC}
    )


target_include_directories(${CMAKE_PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

# adjust this path depending where cmake searches for the target files
set(ConfigPackageLocation "${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}")

install(TARGETS ${CMAKE_PROJECT_NAME}
    ARCHIVE DESTINATION ${ACCRQA_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${ACCRQA_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${ACCRQA_BIN_INSTALL_DIR} COMPONENT runtime
)

if (NOT DEFINED PYTHON_EXECUTABLE)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include"
        DESTINATION ${ACCRQA_INCLUDE_INSTALL_DIR}
        COMPONENT headers
        FILES_MATCHING REGEX "(.*h)|(.*hpp)"
    )
    
    configure_package_config_file(
        ${CMAKE_PROJECT_NAME}Config.cmake.in
        ${CMAKE_PROJECT_NAME}Config.cmake
        INSTALL_DESTINATION "${ConfigPackageLocation}"
        PATH_VARS CMAKE_INSTALL_PREFIX
        )
    
    write_basic_package_version_file(
        ${CMAKE_PROJECT_NAME}ConfigVersion.cmake
        VERSION ${ACCRQA_VERSION_STRING}
        COMPATIBILITY AnyNewerVersion
        )
    
    install(
      FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
      DESTINATION
        ${ConfigPackageLocation}
      COMPONENT
        Development
    )
endif()
