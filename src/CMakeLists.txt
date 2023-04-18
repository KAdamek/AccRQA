
set(${CMAKE_PROJECT_NAME}_CUDA_SRC
    src/AccRQA-GPU-HST.cu
    src/AccRQA-GPU-RR.cu
    PARENT_SCOPE
)

set(${CMAKE_PROJECT_NAME}_SRC
    src/AccRQA_CPU_function.cpp
    src/AccRQA_functions.cpp
    src/AccRQA_utilities_mem.cpp
    src/AccRQA_python_bindings.cpp
    src/AccRQA_R_bindings.cpp
    PARENT_SCOPE
)
