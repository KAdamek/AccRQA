******
AccRQA
******

AccRQA is a multi-platform (CPU, NVIDIA GPUs) application that calculates RQA metrics. Acceleration using GPUs is optional and is not required to have an NVIDIA GPU. AccRQA offers high-performance implementation of RQA metrics that are available through Python, R, and C/C++. 

The interfaces of the AccRQA library are designed platform and number precision agnostic. Change of the computational platform requires a change of a single parameter in the function invocation. If the selected computational platform is unavailable, the AccRQA library automatically switches to CPU. Therefore, transitioning from a locally run workflow (for example, on a laptop) to a more powerful desktop of an HPC cluster with GPUs is effortless. 

Supported platforms are CPUs and NVIDIA GPUs. We are planning to support AMD GPUs in the future.

Interfaces assume that data are resident in the main memory (CPU RAM) and are transferred to the computational platform of the user's choice. 

AccRQA library exposes three functions to the user. These functions calculate:
 - RR
 - LAM, including TT, TTmax
 - DET, including L, Lmax, ENTR

AccRQA library supports Euclidean and maximal norms with plans to provide more in the future.

.. toctree::
   :caption: Installation guide
   :maxdepth: 1
   
   install

.. toctree::
   :caption: Developer guide
   :maxdepth: 1
  
   how_to_contribute

.. toctree::
   :caption: Processing functions
   :maxdepth: 1
   
   rqa_metrics

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
