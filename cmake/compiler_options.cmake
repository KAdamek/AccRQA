macro(APPEND_FLAGS FLAG_VAR)
    foreach (flag ${ARGN})
        set(${FLAG_VAR} "${${FLAG_VAR}} ${flag}")
    endforeach()
endmacro()

# Set general compiler flags.
set(BUILD_SHARED_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT WIN32)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*")
        append_flags(CMAKE_CXX_FLAGS -stdlib=libc++)
    endif()

    if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*"
            OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
        append_flags(CMAKE_C_FLAGS -std=c11)

        append_flags(CMAKE_CXX_FLAGS
            -Wall -Wextra -pedantic -Wcast-qual -Wcast-align -Wno-long-long
            -Wno-variadic-macros
            -fvisibility-inlines-hidden
            -fdiagnostics-show-option)
        append_flags(CMAKE_C_FLAGS
            -Wall -Wextra -pedantic -Wcast-qual -Wcast-align -Wno-long-long
            -Wno-variadic-macros -Wmissing-prototypes
            -fdiagnostics-show-option)
    endif()
endif ()

# Set CUDA releated compiler flags.
# --compiler-options or -Xcompiler: specify options directly to the compiler
#                                   that nvcc encapsulates.
# ------------------------------------------------------------------------------
if (CUDA_FOUND)
    if (MSVC)
        set(CUDA_PROPAGATE_HOST_FLAGS ON)
    else()
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    endif()
    set(CUDA_VERBOSE_BUILD OFF)

    # General NVCC compiler options.
    list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread")
    set(CUDA_NVCC_FLAGS_RELEASE "-O3")
    set(CUDA_NVCC_FLAGS_DEBUG "-O0 -g --generate-line-info")
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "-O3 -g --generate-line-info")
    set(CUDA_NVCC_FLAGS_MINSIZEREL "-O1")
    if (DEFINED NVCC_COMPILER_BINDIR)
        append_flags(CUDA_NVCC_FLAGS -ccbin ${NVCC_COMPILER_BINDIR})
    endif()

    # Options passed to the compiler NVCC encapsulates.
    if (APPLE AND (DEFINED CMAKE_OSX_DEPLOYMENT_TARGET))
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET};)
    endif()
    if (NOT WIN32)
        if (FORCE_LIBSTDC++)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-stdlib=libstdc++;)
        endif()
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fvisibility=hidden;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wextra;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-unused-private-field;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-unused-parameter;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-variadic-macros;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-long-long;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-unused-function;)
        if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*")
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-unused-local-typedef;)
        endif()
        # PTX compiler options
        #list(APPEND CUDA_NVCC_FLAGS_RELEASE --ptxas-options=-v;)
    endif ()

    message("===============================================================================")
    if (NOT DEFINED CUDA_ARCH)
        set(CUDA_ARCH "ALL")
        message("-- INFO: Setting CUDA_ARCH to ALL.")
        message("-- INFO: The target CUDA architecture can be specified using:")
        message("-- INFO:   -DCUDA_ARCH=\"<arch>\"")
        message("-- INFO: where <arch> is one or more of:")
        message("-- INFO:   3.0, 3.5, 3.7, 5.0, 5.2, 6.0, 6.1, 6.2,")
        message("-- INFO:   7.0, 7.5, 8.0, 8.6, 8.7, 9.0 or ALL.")
        message("-- INFO: Note that ALL is currently most from 6.0 to 8.6.")
        message("-- INFO: Separate multiple architectures with semi-colons.")
    endif()
    foreach (ARCH ${CUDA_ARCH})
        if (ARCH MATCHES ALL|[Aa]ll)
            message("-- INFO: Building CUDA device code for supported")
            message("-- INFO: Pascal, Volta, Turing and Ampere architectures:")
            message("-- INFO: 6.0, 6.1, 7.0, 7.5, 8.0, 8.6.")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86)
        elseif (ARCH MATCHES 3.0)
            message("-- INFO: Building CUDA device code for architecture 3.0 (Early Kepler)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
        elseif (ARCH MATCHES 3.5)
            message("-- INFO: Building CUDA device code for architecture 3.5 (K20; K40)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
        elseif (ARCH MATCHES 3.7)
            message("-- INFO: Building CUDA device code for architecture 3.7 (K80)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
        elseif (ARCH MATCHES 5.0)
            message("-- INFO: Building CUDA device code for architecture 5.0 (Early Maxwell)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
        elseif (ARCH MATCHES 5.2)
            message("-- INFO: Building CUDA device code for architecture 5.2 (GTX 9xx-series)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
        elseif (ARCH MATCHES 6.0)
            message("-- INFO: Building CUDA device code for architecture 6.0 (P100)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
        elseif (ARCH MATCHES 6.1)
            message("-- INFO: Building CUDA device code for architecture 6.1 (GTX 10xx-series)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
        elseif (ARCH MATCHES 6.2)
            message("-- INFO: Building CUDA device code for architecture 6.2")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
        elseif (ARCH MATCHES 7.0)
            message("-- INFO: Building CUDA device code for architecture 7.0 (V100; Titan V)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
        elseif (ARCH MATCHES 7.5)
            message("-- INFO: Building CUDA device code for architecture 7.5 (RTX 20xx-series; GTX 16xx-series; Titan RTX)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75)
        elseif (ARCH MATCHES 8.0)
            message("-- INFO: Building CUDA device code for architecture 8.0 (A100)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80)
        elseif (ARCH MATCHES 8.6)
            message("-- INFO: Building CUDA device code for architecture 8.6 (RTX 30xx-series)")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86)
        elseif (ARCH MATCHES 8.7)
            message("-- INFO: Building CUDA device code for architecture 8.7")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_87,code=sm_87)
        elseif (ARCH MATCHES 9.0)
            message("-- INFO: Building CUDA device code for architecture 9.0 (GH100 H200, H100 Hopper")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_90,code=sm_90)
        else()
            message(FATAL_ERROR "-- CUDA_ARCH ${ARCH} not recognised!")
        endif()
    endforeach()
endif (CUDA_FOUND)

# Printing only below this line.
message("===============================================================================")
message("-- INFO: AccRQA version    ${ACCRQA_VERSION_ID}")
message("-- INFO: Build type        ${CMAKE_BUILD_TYPE}")
message("-- INFO: Compiler ID       ${CMAKE_C_COMPILER_ID}:${CMAKE_CXX_COMPILER_ID}")
message("===============================================================================")
message("-- INFO: 'make install' will install the project to:")
message("-- INFO:   - Libraries         ${CMAKE_INSTALL_PREFIX}/${ACCRQA_LIB_INSTALL_DIR}")
message("-- INFO:   - Headers           ${CMAKE_INSTALL_PREFIX}/${ACCRQA_INCLUDE_INSTALL_DIR}")
message("-- NOTE: These paths can be changed using: '-DCMAKE_INSTALL_PREFIX=<path>'")
message("===============================================================================")
if (NOT CUDA_FOUND)
    message("===============================================================================")
    message("-- WARNING: CUDA toolkit not found.")
    message("===============================================================================")
endif()
if (BUILD_INFO)
    message("===============================================================================")
    message(STATUS "C++ compiler  : ${CMAKE_CXX_COMPILER}")
    message(STATUS "C compiler    : ${CMAKE_C_COMPILER}")
    if (DEFINED NVCC_COMPILER_BINDIR)
        message(STATUS "nvcc bindir   : ${NVCC_COMPILER_BINDIR}")
    endif()
    if (FORCE_LIBSTDC++)
        message(STATUS "Forcing linking with libstdc++")
    endif()
    if (${CMAKE_BUILD_TYPE} MATCHES [Rr]elease)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELEASE}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Dd]ebug)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_DEBUG}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Rr]elWithDebInfo)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELWITHDEBINFO}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Mm]inSizeRel)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS}$ {CMAKE_C_FLAGS_MINSIZEREL}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_MINSIZEREL}")
    endif()
    message("===============================================================================")
endif (BUILD_INFO)
