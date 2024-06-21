# AccRQA
GPU accelerated calculation of RQA metrics using CUDA

Documentation
===
The documentation is on GitHub [pages](https://kadamek.github.io/AccRQA/index.html). (https://kadamek.github.io/AccRQA/index.html)

Installation and Usage (Linux)
===

To configure the build system using CMake, create a `build` directory

    mkdir build

and then

    cd build/

run CMake

    cmake ../

The CUDA architecture can be specified with the `-DCUDA_ARCH` flag. For example, for architecture `7.0`, do

    cmake -DCUDA_ARCH="7.0" ../

To compile binding to R, you have to specify R library location using `-DCMAKE_R_LIB_DIR` and location of the R include directory using `-DCMAKE_R_INC_DIR`. For example

    cmake -DCMAKE_R_INC_DIR="/usr/share/R/include" -DCMAKE_R_LIB_DIR="/usr/lib/R/lib/libR.so"  ../

The software can then be compiled using the generated Makefile. To do so, simply type

    make

