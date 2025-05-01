******************
Installation guide
******************

If GPU acceleration is required, make sure the CUDA toolkit is installed first.

The C library
=============

The AccRQA library is compiled from source using CMake. Although, AccRQA can be compiled in Linux and Windows, we strongly suggest using Linux.

Linux
-----

Clone AccRQA repository using:

  .. code-block:: bash

     git clone https://github.com/KAdamek/AccRQA.git

From the top-level directory of the AccRQA, run the following commands to compile and
install the library:

  .. code-block:: bash

     mkdir build
     cd build
     cmake [OPTIONS] ../
     make
     make install

Suitable CUDA architecture is selected by nvcc when compiling because the default option for CUDA is ``-DCUDA_ARCH="native"``. However, if you need to specify the CUDA architecture, use ``-DCUDA_ARCH="x.y"`` flag, where x and y are major and minor compute capability on NVIDIA GPU. 
For example, for architecture `7.0`, do

  .. code-block:: bash
  
     cmake -DCUDA_ARCH="7.0" ../

To compile binding to R, you have to specify R library location using 
``-DCMAKE_R_LIB_DIR`` and location of the R include directory using 
``-DCMAKE_R_INC_DIR``. For example

  .. code-block:: bash

     cmake -DCMAKE_R_INC_DIR="/usr/share/R/include" -DCMAKE_R_LIB_DIR="/usr/lib/R/lib/libR.so"  ../

To compile tests use ``-DBUILD_TESTS=ON`` which compiles test executable performing a series of tests of supported RQA metrics.

An example application that uses the AccRQA library can be compiled with a flag ``-DBUILD_APPLICATIONS=ON``.

Windows
-------

To compile the AccRQA library from source you will require:
  - MSVC compiler (part of Visual Studio)
  - Make for Windows


In command line you can build AccRQA by

  .. code-block:: bash

     mkdir build
     cd build
     cmake .. [OPTIONS]
     cmake --build .
     cmake --install .

Python package
==============

Install from source package
---------------------------

We encourage you to install the accrqa package from the source package, as Python binary wheels poorly support GPUs.

  .. code-block:: bash

     pip install accrqa

Install from the binary wheel
-----------------------------

Working on it.

Install from the local repository
---------------------------------

From the top-level directory, run the following commands to install
the Python package:

  .. code-block:: bash

     pip3 install .

The compiled library will be built as part of this step, so it does not need to
be installed separately. If extra CMake arguments need to be specified, set the
environment variable ``CMAKE_ARGS`` first, for example:

  .. code-block:: bash

     CMAKE_ARGS="-DCUDA_ARCH=7.0" pip3 install .


Uninstalling
------------

The Python package can be uninstalled using:

  .. code-block:: bash

     pip3 uninstall accrqa

Installing on Windows
---------------------

To install accrqa package on Windows the easiest way is to use the binary wheels. To install accrqa Python package on Windows from source you need to have 

Windows requirements:
  - MSVC compiler (part of Visual Studio)
  - Make for Windows

To install from PiPy using pip:

  .. code-block:: bash

     py -m pip install accrqa

To install from the local repository in command line:

  .. code-block:: bash

     py -m pip install .