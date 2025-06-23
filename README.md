# AccRQA
AccRQA is a multi-platform (CPU, NVIDIA GPUs) application that calculates RQA metrics. Acceleration using GPUs is optional and is not required to have an NVIDIA GPU. AccRQA offers high-performance implementation of RQA metrics that are available through Python, R, and C/C++. 

The interfaces of the AccRQA library are designed platform and number precision agnostic. Change of the computational platform requires a change of a single parameter in the function invocation. If the selected computational platform is unavailable, the AccRQA library automatically switches to CPU. Therefore, transitioning from a locally run workflow (for example, on a laptop) to a more powerful desktop of an HPC cluster with GPUs is effortless. 

Supported platforms are CPUs and NVIDIA GPUs. We are planning to support AMD GPUs in the future.

Interfaces assume that data are resident in the main memory (CPU RAM) and are transferred to the computational platform of the user's choice. 

AccRQA library exposes three functions to the user. These functions calculate:
 - RR
 - LAM, including TT, TTmax
 - DET, including L, Lmax, ENTR

AccRQA library supports Euclidean and maximal norms with plans to provide more in the future.

Documentation
===
The documentation is on GitHub [pages](https://kadamek.github.io/AccRQA/index.html). (https://kadamek.github.io/AccRQA/index.html)

How to install Python package
===

We encourage you to install the accrqa package from the source package, as Python binary wheels poorly support GPUs.

    pip install accrqa

Binary wheels can be downloaded from GitHub release page.

The Python package can be uninstalled using:

     pip3 uninstall accrqa

Install from local repository
---

From the top-level directory, run the following commands to install
the Python package:

     pip3 install .

The compiled library will be built as part of this step, so it does not need to
be installed separately. If extra CMake arguments need to be specified, set the
environment variable ``CMAKE_ARGS`` first, for example:

     CMAKE_ARGS="-DCUDA_ARCH=7.0" pip3 install .

Installing on Windows
---

To install accrqa Python package on Windows from source you need to have 

Windows requirements:
  - MSVC compiler (part of Visual Studio)
  - Make for Windows

To install from PiPy using pip:

     py -m pip install accrqa

To install from the local repository in command line:

     py -m pip install .


How to Install the R Package
===


### System Requirements

To install and use the AccRQA R package, the following system requirements apply:

- **R version**: 4.0 or newer.
- **Operating system**: Linux, or Windows.
- **Build tools**:
  - On **Linux/macOS**: `gcc`, `g++`, `make` (usually provided by `build-essential`).
  - On **Windows**: [Rtools](https://cran.r-project.org/bin/windows/Rtools/).
- *(Optional)* **CUDA Toolkit** (e.g., 11.x or newer) — required for GPU acceleration.
  - Ensure `nvcc` and `cuda_runtime.h` are available in the environment.
  - Set `CUDA_HOME` if the toolkit is not in the default path.

You can install the **AccRQA R package** either from a downloaded source archive (e.g., `.tar.gz` or `.zip`) or directly within RStudio.

### Option 1: Install from `.zip` or `.tar.gz` in RStudio

1. Open **RStudio**.
2. Go to **Tools → Install Packages**.
3. From the box Install from select **"Package Archive File (.tar.gz / .zip)"** as the sources.
4. Click **Browse**, and select the downloaded file (e.g., `AccRQA_x.y.z.tar.gz` where the `x.y.z` refers to the specific version number).
5. Click **Install**.

> Make sure that Rtools (on Windows) or build-essential (on Linux) is installed to compile packages from source.

### Option 2: Install from the Command Line

The downloaded archive can be install by using the R console:
```
r

# From inside R
install.packages("AccRQA_x.y.z.tar.gz", repos = NULL, type = "source")
```
or from your system's terminal:
````
bash

R CMD INSTALL AccRQA_x.y.z.tar.gz
````

How to Install the C library (Linux)
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

To compile tests set `-DBUILD_TESTS=ON`.
To compile example C/C++ application set `-DBUILD_APPLICATIONS=ON`.

The software can then be compiled using the generated Makefile. To do so, simply type

    make

To compile tests use ``-DBUILD_TESTS=ON`` which compiles test executable performing a series of tests of supported RQA metrics.

An example application that uses the AccRQA library can be compiled with a flag ``-DBUILD_APPLICATIONS=ON``.


Installation of C library (Windows)
---

To compile the AccRQA library from source you will require:
  - MSVC compiler (part of Visual Studio)
  - Make for Windows


In command line you can build AccRQA by

    mkdir build
    cd build
    cmake .. [OPTIONS]
    cmake --build .
    cmake --install .


Contributors
===
 - Karel Adamek
 - Jan Novotny
 - Radim Panis
 - Norbert Marwan
 
