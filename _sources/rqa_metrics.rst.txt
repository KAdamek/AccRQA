****************
AccRQA functions
****************

C/C++
=====

.. doxygengroup:: rqa_metrics
   :content-only:

Python
======

.. autofunction:: accrqa.RP

.. autofunction:: accrqa.RR

.. autofunction:: accrqa.LAM

.. autofunction:: accrqa.DET

.. autofunction:: accrqa.RR_target

.. autofunction:: accrqa.accrqaDistance

.. autofunction:: accrqa.accrqaCompPlatform

R
=

.. function::    accrqa_RP(input_data, tau, emb, threshold, distance_type = "euclidean")

Calculates recurrence plot from supplied time-series and return the RP plot.

The choice of ``threshold`` strongly controls the recurrence rate (RR).
Smaller thresholds produce sparser plots (low RR), larger thresholds denser
plots (high RR). Typical practice is to choose ``threshold`` such that
RR is in a reasonable range (e.g. 1-5-10\%) and then compute DET, LAM, ENTR
on the resulting RP.

The returned RP is an NxN matrix, where N is the length
of the (possibly embedded) time series. It can be visualised with
``plot}()`` or your own plotting routines.

:Parameters: ``input_data`` : Numeric vector
        Numeric vector with the time series.

    ``tau`` : numeric
        Delay (integer, scalar).

    ``emb`` : numeric
        Embedding dimension (integer, scalar).

    ``threshold`` : numeric
        Threshold for recurrence (numeric, scalar).

    ``distance_type`` : character string
        Character string specifying distance: one of "euclidean" or "maximal".

:Returns:
    `class` **accrqa_rp** containing:
    
        * **output**: Integer vector of length \code{rp_size^2} with RP (0/1).
        * **input**: Original input data (numeric vector).
        * **input_size**: Length of the input series.
        * **tau**: Delay (integer).
        * **emb**: Embedding dimension (integer).
        * **threshold**: Threshold used.
        * **distance_type**: Distance type as character.
        * **rp_size**: Effective RP side length after embedding.

:Example:

.. code-block:: r

   ts <- sin(2 * pi * (1:100) / 20)
   rp <- accrqa_RP(ts, tau = 1, emb = 2, threshold = 0.5, distance_type = "euclidean")
   plot(rp)
   plot(rp, summary = TRUE)



.. function::    accrqa_RR(input_data, tau_values, emb_values, threshold_values, distance_type = "euclidean", comp_platform)

This function computes the recurrence rate (RR) for a given input time series based on the specified
delays, embedding dimensions, and thresholds. The function allows the user to specify normalization
and computational platform.

Recurrence rate (RR) quantifies the density of recurrence points in a recurrence plot.
This function uses a compiled C backend to efficiently compute RR based on the input parameters.
It performs validations on input lengths and ensures that parameters like delays and embeddings
are integers.

:Parameters:

  ``input_data`` : numeric vector
      A numeric vector representing the input time series.
  
  ``tau_values`` : numeric vector
      A numeric vector of time delay (τ) values.
  
  ``emb_values`` : numeric vector
      A numeric vector of embedding dimension (m) values.
  
  ``threshold_values`` : numeric vector
      A numeric vector of threshold values for recurrence detection.
  
  ``distance_type`` : character string, optional
      A character string specifying the normalization method. Defaults to ``"euclidean"``.
      Possible values are:
      
      * ``"euclidean"``: Euclidean distance.
      * ``"maximal"``: Maximum norm (Chebyshev distance).
  
  ``comp_platform`` : character string
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
   input <- runif(100)
   tau_values <- c(1, 2)
   emb_values <- c(2, 3)
   threshold_values <- c(0.1, 0.2)
   norm <- "euclidean"
   platform <- "cpu"
   result <- accrqa_RR(input, tau_values, emb_values, threshold_values, norm, platform)
   print(result)



.. function::    accrqa_LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type = "euclidean", calculate_ENTR = TRUE, comp_platform)

This function computes laminarity (LAM) based on the given input time series and RQA parameters.

Laminarity (LAM) is a measure in recurrence quantification analysis that describes the tendency of points
to form vertical lines in the recurrence plot. This function provides configurable parameters for
calculating LAM with options for normalization and entropy computation.

:Parameters:
    ``input_data`` : numeric vector
        A numeric vector representing the input time series data.
    
    ``tau_values`` : numeric vector
        A numeric vector of time delay values.
    
    ``emb_values`` : numeric vector
        A numeric vector of embedding dimension values.
    
    ``vmin_values`` : numeric vector
        A numeric vector of minimum vertical line lengths.
    
    ``threshold_values`` : numeric vector
        A numeric vector of threshold values for recurrence detection.
    
    ``distance_type`` : character string, optional
        A character string specifying the distance norm to use. Possible values are:
        
        * ``"euclidean"``: Euclidean distance.
        * ``"maximal"``: Maximum norm (Chebyshev distance).
        * ``"none"``: No normalization.
        
        Default is ``"euclidean"``.
    
    ``calculate_ENTR`` : logical, optional
        A logical value indicating whether to calculate entropy (``TRUE`` or ``FALSE``). Default is ``TRUE``.
    
    ``comp_platform`` : character string
        A character string specifying the computing platform. Options may include
        ``"cpu"``, ``"nv_gpu"``, etc.

:Returns:
  data.frame
      A data frame with the following columns:
      
      * **LAM**: Laminarity percentage.
      * **V**: Mean vertical line length.
      * **Vmax**: Maximum vertical line length.
      * **ENTR**: Entropy of the vertical line length distribution (if ``calculate_ENTR = TRUE``).

:Example:

.. code-block:: r

   # Example usage of accrqa_LAM
   input <- runif(100)
   tau_values <- c(1, 2)
   emb_values <- c(2, 3)
   vmin_values <- c(2, 3)
   threshold_values <- c(0.1, 0.2)
   norm <- "euclidean"
   calculate_ENTR <- TRUE
   comp_platform <- "cpu"
   result <- accrqa_LAM(
     input            = input,
     tau_values       = tau_values,
     emb_values       = emb_values,
     vmin_values      = vmin_values,
     threshold_values = threshold_values,
     distance_type    = norm,
     calculate_ENTR   = calculate_ENTR,
     comp_platform    = comp_platform
   )



.. function::    accrqa_DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type = "euclidean", calculate_ENTR = TRUE, comp_platform)

This function calculates the determinism (DET) for cross-recurrence quantification analysis (CRQA)
based on a set of input parameters, including time delay, embedding dimensions, minimum line length,
threshold values, and normalization.

The function performs cross-recurrence quantification analysis (CRQA) using the specified parameters.
DET measures the proportion of recurrent points forming diagonal lines in the recurrence plot,
which indicates deterministic structures in the data. If ``calculate_ENTR`` is ``TRUE``, the entropy of diagonal
line lengths is also computed.

:Parameters: ``input_data`` : **numeric matrix or data frame**
        A numeric matrix or data frame representing the input data for CRQA analysis.

    ``tau_values`` : **numeric vector**
        A numeric vector specifying the time delay(s) to be used in the analysis.

    ``emb_values`` : numeric vector
        A numeric vector specifying the embedding dimensions to be tested.

    ``lmin_values`` : numeric vector
        A numeric vector specifying the minimum diagonal line lengths for DET computation.

    ``threshold_values`` : numeric vector
        A numeric vector specifying the threshold values for recurrence computation.

    ``distance_type`` : character string, optional
        A character string specifying the normalization method to be used. Options may include ``"euclidean"``, ``"maximal"``, etc. Default is ``"euclidean"``.

    ``calculate_ENTR`` : logical, optional
        A logical value indicating whether to calculate entropy (ENTR) along with DET. Default is ``TRUE``.

    ``comp_platform`` : character string
        A character string specifying the computing platform. Options may include ``"cpu"``, ``"nv_gpu"``, etc.

:Returns:
    `data.frame`
        A data frame containing:
    
        * **Delay**: Specific time delay from the values set in the parameters.
        * **Embedding**: Specific embedding dimension from the values set in the parameters.
        * **Lmin**: Minimal diagonal line lengths set for DET computation.
        * **DET**: The determinism values computed for the given input parameters.
        * **ENTR** (if ``calculate_ENTR = TRUE``): The entropy values corresponding to the DET computations.
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
     input_data = input_data,
     tau_values = tau,
     emb_values = emb,
     lmin_values = lmin,
     threshold_values = threshold,
     distance_type = norm_method,
     calculate_ENTR = calculate_entropy,
     comp_platform = comp_platform
   )



.. function::    accrqa_RR_target(input_data, tau, emb, target_RR, epsilon = 0.01, distance_type = "euclidean", comp_platform = "cpu", max_iter = 20, threshold_min = 0, threshold_max = NULL, n_threshold = 10, verbose = FALSE, ...)

Find threshold that yields target RR (recurrence rate)

:Parameters: ``input_data`` : **Numeric vector**
        A numeric matrix or data frame representing the input data for CRQA analysis.

    ``tau`` : numeric
        Delay (integer, scalar).

    ``emb`` : numeric
        Embedding dimension (integer, scalar).

    ``target_RR`` : numeric vector
        A numeric vector specifying the minimum diagonal line lengths for DET computation.

    ``epsilon`` : numeric
        Allowed error to find the desired RR. Default: 0.01

    ``distance_type`` : character string, optional
        A character string specifying the normalization method to be used. Options may include ``"euclidean"``, ``"maximal"``, etc. Default is ``"euclidean"``.

    ``comp_platform`` : character string
        A character string specifying the computing platform. Options may include ``"cpu"``, ``"nv_gpu"``, etc.

    ``max_iter`` : numeric
        Maximum iterations to find the threshold before give up. Default: 20

    ``threshold_min`` : numeric
        Lower bound for search. Default: 0

    ``threshold_max`` : numeric
        Upper bound for search (set automatically if NULL). Default: NULL

    ``n_threshold`` : numeric
        How many thresholds per iteration (e.g. 10). Default: 10

    ``verbose`` : binary
        If TRUE, prints progress. Default: FALSE

    ``...`` : 
        Further arguments (currently ignored but kept for S3 compatibility).

:Returns:
    `data.frame`
        A data frame containing:
    
        * **target_RR**: Target RR.
        * **threshold**: Threshold to achieve desired RR.
        * **RR_found**: RR at the found threshold.

:Example:

.. code-block:: r

   x <- runif(1000)
   accrqa_RR_target(x, tau = 1, emb = 2, target_RR = 0.5, epsilon = 0.001)
   # multiple targets
   accrqa_RR_target(x, tau = 1, emb = 2, target_RR = c(0.1,0.5,0.8), epsilon = 0.001)