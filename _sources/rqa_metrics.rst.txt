****************
AccRQA functions
****************

C/C++
=====

.. doxygengroup:: rqa_metrics
   :content-only:

Python
======

.. autofunction:: accrqa.RR

.. autofunction:: accrqa.LAM

.. autofunction:: accrqa.DET

.. autofunction:: accrqa.accrqaDistance

.. autofunction:: accrqa.accrqaCompPlatform

R
=

.. function::    accrqa_DET(input, tau_values, emb_values, lmin_values, threshold_values, norm = "euclidean", calc_ENTR = TRUE, platform)

This function calculates the determinism (DET) for cross-recurrence quantification analysis (CRQA)
based on a set of input parameters, including time delay, embedding dimensions, minimum line length,
threshold values, and normalization.

The function performs cross-recurrence quantification analysis (CRQA) using the specified parameters.
DET measures the proportion of recurrent points forming diagonal lines in the recurrence plot,
which indicates deterministic structures in the data. If ``calc_ENTR`` is ``TRUE``, the entropy of diagonal
line lengths is also computed.

:Parameters: `input` : **numeric matrix or data frame**
        A numeric matrix or data frame representing the input data for CRQA analysis.

    `tau_values` : **numeric vector**
        A numeric vector specifying the time delay(s) to be used in the analysis.

    `emb_values` : numeric vector
        A numeric vector specifying the embedding dimensions to be tested.

    `lmin_values` : numeric vector
        A numeric vector specifying the minimum diagonal line lengths for DET computation.

    `threshold_values` : numeric vector
        A numeric vector specifying the threshold values for recurrence computation.

    `norm` : character string, optional
        A character string specifying the normalization method to be used. Options may include ``"euclidean"``, ``"maximal"``, etc. Default is ``"euclidean"``.

    `calc_ENTR` : logical, optional
        A logical value indicating whether to calculate entropy (ENTR) along with DET. Default is ``TRUE``.

    `platform` : character string
        A character string specifying the computing platform. Options may include ``"cpu"``, ``"nv_gpu"``, etc.

:Returns:
    `data.frame`
        A data frame containing:
    
        * **Delay**: Specific time delay from the values set in the parameters.
        * **Embedding**: Specific embedding dimension from the values set in the parameters.
        * **Lmin**: Minimal diagonal line lengths set for DET computation.
        * **DET**: The determinism values computed for the given input parameters.
        * **ENTR** (if ``calc_ENTR = TRUE``): The entropy values corresponding to the DET computations.
        * **RR**: RR values.

:Example:

.. code-block:: r

   # Example usage
   input_data <- matrix(runif(100), nrow = 10)
   tau <- c(1, 2)
   emb <- c(2, 3)
   lmin <- 1
   threshold <- 1
   norm_method <- "euclidean"
   calculate_entropy <- TRUE
   comp_platform <- "cpu"
   
   results <- accrqa_DET(
     input = input_data,
     tau_values = tau,
     emb_values = emb,
     lmin_values = lmin,
     threshold_values = threshold,
     norm = norm_method,
     calc_ENTR = calculate_entropy,
     platform = comp_platform
   )


.. function::    accrqa_LAM(input, tau_values, emb_values, vmin_values, threshold_values, norm = "euclidean", calc_ENTR = TRUE, platform)

This function computes laminarity (LAM) based on the given input time series and RQA parameters.

Laminarity (LAM) is a measure in recurrence quantification analysis that describes the tendency of points
to form vertical lines in the recurrence plot. This function provides configurable parameters for
calculating LAM with options for normalization and entropy computation.

:Parameters:
    `input` : numeric vector
        A numeric vector representing the input time series data.
    
    `tau_values` : numeric vector
        A numeric vector of time delay values.
    
    `emb_values` : numeric vector
        A numeric vector of embedding dimension values.
    
    `vmin_values` : numeric vector
        A numeric vector of minimum vertical line lengths.
    
    `threshold_values` : numeric vector
        A numeric vector of threshold values for recurrence detection.
    
    `norm` : character string, optional
        A character string specifying the distance norm to use. Possible values are:
        
        * ``"euclidean"``: Euclidean distance.
        * ``"maximal"``: Maximum norm (Chebyshev distance).
        * ``"none"``: No normalization.
        
        Default is ``"euclidean"``.
    
    `calc_ENTR` : logical, optional
        A logical value indicating whether to calculate entropy (``TRUE`` or ``FALSE``). Default is ``TRUE``.
    
    `platform` : character string
        A character string specifying the computing platform. Options may include
        ``"cpu"``, ``"nv_gpu"``, etc.

:Returns:
  data.frame
      A data frame with the following columns:
      
      * **LAM**: Laminarity percentage.
      * **V**: Mean vertical line length.
      * **Vmax**: Maximum vertical line length.
      * **ENTR**: Entropy of the vertical line length distribution (if ``calc_ENTR = TRUE``).

:Example:

.. code-block:: r

   # Example usage of accrqa_LAM
   input <- c(1.0, 2.0, 3.0, 4.0)
   tau_values <- c(1, 2)
   emb_values <- c(2, 3)
   vmin_values <- c(2, 3)
   threshold_values <- c(0.1, 0.2)
   norm <- "euclidean"
   calc_ENTR <- TRUE
   
   result <- accrqa_LAM(
     input, 
     tau_values, 
     emb_values, 
     vmin_values, 
     threshold_values, 
     norm, 
     calc_ENTR
   )


.. function::    accrqa_RR(input, tau_values, emb_values, threshold_values, norm = "euclidean", platform)

This function computes the recurrence rate (RR) for a given input time series based on the specified
delays, embedding dimensions, and thresholds. The function allows the user to specify normalization
and computational platform.

Recurrence rate (RR) quantifies the density of recurrence points in a recurrence plot.
This function uses a compiled C backend to efficiently compute RR based on the input parameters.
It performs validations on input lengths and ensures that parameters like delays and embeddings
are integers.

:Parameters:

  `input` : numeric vector
      A numeric vector representing the input time series.
  
  `tau_values` : numeric vector
      A numeric vector of time delay (τ) values.
  
  `emb_values` : numeric vector
      A numeric vector of embedding dimension (m) values.
  
  `threshold_values` : numeric vector
      A numeric vector of threshold values for recurrence detection.
  
  `norm` : character string, optional
      A character string specifying the normalization method. Defaults to ``"euclidean"``.
      Possible values are:
      
      * ``"euclidean"``: Euclidean distance.
      * ``"maximal"``: Maximum norm (Chebyshev distance).
  
  `platform` : character string
      A character string specifying the computational platform. Possible values are:
      
      * ``"cpu"``: Use the CPU for computations.
      * ``"nv_gpu"``: Use an NVIDIA GPU for computations.

:Returns:

  `data.frame`
      A data frame containing:
      
      * **Delay**: The delay (τ) values used in the computation.
      * **Embedding**: The embedding dimension (m) values used.
      * **Threshold**: The threshold values used.
      * **RR**: The computed recurrence rate for each combination of parameters.

:Example:

.. code-block:: r

   # Example usage of accrqa_RR
   input <- c(1.0, 2.0, 3.0, 4.0)
   tau_values <- c(1, 2)
   emb_values <- c(2, 3)
   threshold_values <- c(0.1, 0.2)
   norm <- "euclidean"
   platform <- "cpu"
   
   result <- accrqa_RR(
     input, 
     tau_values, 
     emb_values, 
     threshold_values, 
     norm, 
     platform
   )
   print(result)

