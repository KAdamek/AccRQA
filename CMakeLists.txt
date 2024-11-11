cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(accrqa CXX)
set(ACCRQA_VERSION_ID "0x000001")
set(ACCRQA_VERSION_MAJOR 0)
set(ACCRQA_VERSION_MINOR 1)
set(ACCRQA_VERSION_PATCH 0)
set(ACCRQA_VERSION_SUFFIX "")

set(CMAKE_CXX_STANDARD 11)

option(BUILD_APPLICATIONS "Build example C/C++ application" OFF)
option(BUILD_TESTS "Build test program" OFF)

# Add local CMake module path.
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)

# Get string length of current source directory for log messages.
string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions(-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE})

# Set install directory paths.
if (NOT ACCRQA_LIB_INSTALL_DIR)
    set(ACCRQA_LIB_INSTALL_DIR "lib")
endif()
if (NOT ACCRQA_INCLUDE_INSTALL_DIR)
    set(ACCRQA_INCLUDE_INSTALL_DIR "include")
endif()
if (NOT ACCRQA_BIN_INSTALL_DIR)
    set(ACCRQA_BIN_INSTALL_DIR "bin")
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
find_package(OpenMP QUIET)

# Set compiler options.
include(compiler_options)
enable_testing()

# Generate source list from subdirectories.
add_subdirectory(src)

# Build library using list of source files.
if (CUDAToolkit_FOUND)
    # Add CUDA kernel sources.
    list(APPEND ${CMAKE_PROJECT_NAME}_SRC ${${CMAKE_PROJECT_NAME}_CUDA_SRC})
    add_library(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_SRC})
    target_link_libraries(${CMAKE_PROJECT_NAME} CUDA::cudart_static)
else()
    add_library(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_SRC})
endif()

if (NOT DEFINED PYTHON_EXECUTABLE)
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
        SOVERSION ${ACCRQA_VERSION_ID}
        VERSION ${ACCRQA_VERSION_ID})
else()
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

# Build example application.
if(BUILD_APPLICATIONS)
    add_subdirectory(application)
endif()

# Build tests.
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

#target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_CUFFT_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} OpenMP::OpenMP_CXX)

install(TARGETS ${CMAKE_PROJECT_NAME}
    ARCHIVE DESTINATION ${ACCRQA_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${ACCRQA_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${ACCRQA_BIN_INSTALL_DIR} COMPONENT runtime
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/include"
    DESTINATION ${ACCRQA_INCLUDE_INSTALL_DIR}
    COMPONENT headers
    FILES_MATCHING REGEX "(.*h)|(.*hpp)"
)