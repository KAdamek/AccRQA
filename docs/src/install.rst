******************
Installation guide
******************

If GPU acceleration is required, make sure the CUDA toolkit is installed first.

The C library
=============

The AccRQA library is compiled from source using CMake.

From the top-level directory, run the following commands to compile and
install the library:

  .. code-block:: bash

     mkdir build
     cd build
     cmake .. [OPTIONS]
     make
     make install

To specify the CUDA architecture use ``-DCUDA_ARCH="x.y"`` flag, where x and y 
are major and minor compute capability on NVIDIA GPU. 
For example, for architecture `7.0`, do

  .. code-block:: bash
  
     cmake -DCUDA_ARCH="7.0" ../

To compile binding to R, you have to specify R library location using 
``-DCMAKE_R_LIB_DIR`` and location of the R include directory using 
``-DCMAKE_R_INC_DIR``. For example

  .. code-block:: bash

     cmake -DCMAKE_R_INC_DIR="/usr/share/R/include" -DCMAKE_R_LIB_DIR="/usr/lib/R/lib/libR.so"  ../

The Python library
==================

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
