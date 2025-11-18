# AccRQA
AccRQA is a multi-platform (CPU, NVIDIA GPUs) application that calculates RQA measures. Acceleration using GPUs is optional and is not required to have an NVIDIA GPU to use AccRQA. AccRQA offers high-performance implementation of RQA measures that are available through Python, R, and C/C++. 

The interfaces of the AccRQA library are designed platform and number precision agnostic. Change of the computational platform requires a change of a single parameter in the function invocation. If the selected computational platform is unavailable, the AccRQA library automatically switches to CPU. Therefore, transitioning from a locally run workflow (for example, on a laptop) to a more powerful desktop or an HPC cluster with GPUs is effortless. 

Supported platforms are CPUs and NVIDIA GPUs. We are planning to support AMD GPUs in the future.

Interfaces assume that data are resident in the main memory (CPU RAM) and are transferred to the computational platform of the user's choice by AccRQA. 

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

We encourage you to install the accrqa package from the source package, as Python binary wheels do not support GPUs well and accrqa package will be compiled directly for GPU architecture you have. For the installation you will require an compiler installed (GCC for linux or MSVC for Windows), CUDA toolkit if GPU acceleration is desired and setuptools 63.1.0 or newer. To install accrqa from source use:

    pip install accrqa

Install from the binary wheel
---

Binary wheels can be downloaded from GitHub release page. To install accrqa from binary wheel use:

     pip3 install accrqa_version_.whl

Install from the local repository
---

To install accrqa from local repository, clone AccRQA repository using:

     git clone https://github.com/KAdamek/AccRQA.git

From the top-level directory, run the following commands to install
the Python package:

     pip3 install .

The compiled library will be built as part of this step, so it does not need to
be installed separately. If extra CMake arguments need to be specified, set the
environment variable ``CMAKE_ARGS`` first, for example:

     CMAKE_ARGS="-DCUDA_ARCH=7.0" pip3 install .

Uninstalling
---

The Python package can be uninstalled using:

     pip3 uninstall accrqa


Installing on Windows
---

To install accrqa package on Windows the easiest way is to use the binary wheels. Binary wheels can be downloaded from GitHub release page. To install accrqa using binary wheels use:

     py -m pip install accrqa_version_.whl


To install accrqa Python package on Windows from source you need to have:
  - MSVC compiler (part of Visual Studio)
  - CMake for Windows

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

The AccRQA library can be compiled from source using CMake. To install and use AccRQA you need following:
- **Operating system**: Linux or Windows 10+
- **Build tools**: ``CMake`` 3.13 or newer 
- **Compiler**: ``GCC``  8.1 or newer for linux or ``MSVC`` 2019 or newer for Windows 10
- *(Optional)* **CUDA Toolkit** (e.g., 11.x or newer) — required for GPU acceleration. 

To install git you can go to https://git-scm.com/downloads and follow instruction from there. If GPU acceleration is required, make sure the CUDA toolkit is installed first.

Linux
---

Clone AccRQA repository using:

     git clone https://github.com/KAdamek/AccRQA.git

From the top-level directory of the AccRQA, run the following commands to compile and
install the library:

     mkdir build
     cd build
     cmake [OPTIONS] ../
     make
     make install

Suitable CUDA architecture is selected by nvcc when compiling because the default option for CUDA is ``-DCUDA_ARCH="native"``. However, if you need to specify the CUDA architecture, use ``-DCUDA_ARCH="x.y"`` flag, where x and y are major and minor compute capability on NVIDIA GPU. 
For example, for architecture `7.0`, use

     cmake -DCUDA_ARCH="7.0" ../

To compile tests use ``-DBUILD_TESTS=ON`` which compiles test executable performing a series of tests of supported RQA measures.

An example application that uses the AccRQA library can be compiled with a flag ``-DBUILD_APPLICATIONS=ON``.


Windows
---

To compile the AccRQA library from source on Windows 10 you will require:
- **Build tools**: ``CMake`` 3.13 or newer 
- **Compiler**: ``MSVC`` 2019 or newer


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
 
