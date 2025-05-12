#dyn.load("libAccRQA_R.so")
#' @useDynLib AccRQA, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL


switch_norm <- function(norm){
  output <- switch(
    norm,
    "euclidean" = 1,
    "maximal" = 2,
    0  # Default case
  )
  return(output)
}

switch_platform <- function(platform){
  output <- switch(
    platform,
    "cpu" = 1,
    "nv_gpu" = 1024,
    0
  )
  return(output)
}


#' Calculate Determinism for Cross-Recurrence Quantification Analysis
#'
#' This function calculates the determinism (DET) for cross-recurrence quantification analysis (CRQA)
#' based on a set of input parameters, including time delay, embedding dimensions, minimum line length,
#' threshold values, and normalization.
#'
#' @param input A numeric matrix or data frame representing the input data for CRQA analysis.
#' @param tau_values A numeric vector specifying the time delay(s) to be used in the analysis.
#' @param emb_values A numeric vector specifying the embedding dimensions to be tested.
#' @param lmin_values A numeric vector specifying the minimum diagonal line lengths for DET computation.
#' @param threshold_values A numeric vector specifying the threshold values for recurrence computation.
#' @param norm A character string specifying the normalization method to be used. Options may include
#'   `"euclidean"`, `"maximal"`, etc.
#' @param calc_ENTR A logical value indicating whether to calculate entropy (ENTR) along with DET.
#' @param comp_platform A character string specifying the computing platform. Options may include
#'   `"cpu"`, `"nv_gpu"`, etc.
#' @return A data frame containing:
#'   - `Delay`: Specific time delay from the values set in the parameters.
#'   - `Embedding`: Specific embedding dimension from the values set in the parameters.
#'   - `Lmin`: Minimal diagonal line lengths set for DET computation.
#'   - `DET`: The determinism values computed for the given input parameters.
#'   - `ENTR` (if `calc_ENTR = TRUE`): The entropy values corresponding to the DET computations.
#'   - `RR`: RR values.
#'
#' @details
#' The function performs cross-recurrence quantification analysis (CRQA) using the specified parameters.
#' DET measures the proportion of recurrent points forming diagonal lines in the recurrence plot,
#' which indicates deterministic structures in the data. If `calc_ENTR` is `TRUE`, the entropy of diagonal
#' line lengths is also computed.
#'
#' @examples
#' # Example usage
#' input_data <- matrix(runif(100), nrow = 10)
#' tau <- c(1, 2)
#' emb <- c(2, 3)
#' lmin <- 1
#' threshold <- 1
#' norm_method <- "euclidean"
#' calculate_entropy <- TRUE
#' comp_platform <- "cpu"
#'
#' results <- accrqa_DET(
#'   input = input_data,
#'   tau_values = tau,
#'   emb_values = emb,
#'   lmin_values = lmin,
#'   threshold_values = threshold,
#'   norm = norm_method,
#'   calc_ENTR = calculate_entropy,
#'   platform = comp_platform,
#' )
#'
#' @export
accrqa_DET <- function(input, tau_values, emb_values, lmin_values, threshold_values, norm="euclidean", calc_ENTR=TRUE, platform)
{
  variables <- list(input = input, tau = tau_values, emb = emb_values, lmin = lmin_values, threshold = threshold_values)
  empty_vars <- names(variables)[sapply(variables, function(x) length(x) == 0)]
  
  input_size <- length(input)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nLmins <- length(lmin_values)
  nThresholds <- length(threshold_values)
  output_size <- nTaus*nEmbs*nLmins*nThresholds*5

  if (length(empty_vars) > 0) {
    stop(paste("Number of delays, embedding, minimal lengths or thresholds must be greater than zero or the input frame. The following are empy or null: ", paste(empty_vars, collapse = ", ")))
  }
  
  if( any(variables$tau %% 1 != 0) == TRUE){
    warning("The delay values should be integers only, converting.")
  }
  
  if( any(variables$emb %% 1 != 0) == TRUE){
    warning("The tau values should be integers only, converting.")
  }
  
  if(is.logical(calc_ENTR) == FALSE){
    stop("Invalid value of calculate_ENTR. Should be TRUE or FALSE")
  }
  
  norm_method <- switch_norm(norm)
  comp_platform <- switch_platform(platform)
  
  if (norm_method == 0) {
    stop("Normalization method to be used not recognized. Please use 'euclidean' or 'maximal'.")
  }
  
  if (comp_platform == 0) {
    stop("Platform to compute not recognized. Please use 'cpu' or 'nv_gpu'.")
  }
  
  rst <- .C("R_double_accrqa_DET",
    output = double(length=output_size),
    input = as.double(input),
    input_size = as.integer(input_size),
    tau = as.integer(tau_values),
    tau_size = as.integer(nTaus),
    emb = as.integer(emb_values),
    emb_size = as.integer(nEmbs),
    lmin = as.integer(lmin_values),
    lmin_size = as.integer(nLmins),
    threshold = as.double(threshold_values),
    thr_size = as.integer(nThresholds),
    norm = as.integer(norm_method),
    entr = as.integer(calc_ENTR),
    platform = as.integer(comp_platform)
  )
  
  tidy_df <- expand.grid(
    Threshold = rst$threshold,
    Embedding = rst$emb, 
    Lmin = rst$lmin,
    Delay = rst$tau
  )
  tidy_df <- tidy_df[,c("Delay", "Embedding", "Lmin", "Threshold")]
  
  metrics <- as.data.frame(matrix(rst$output, ncol = 5, byrow = TRUE))
  colnames(metrics) <- c("DET", "L", "Lmax", "ENTR", "RR")
  
  result <- cbind(tidy_df, metrics)
  return(result)
}

#' Calculate Laminarity in Recurrence Quantification Analysis (RQA)
#'
#' This function computes laminarity (LAM) based on the given input time series and RQA parameters.
#'
#' @param input A numeric vector representing the input time series data.
#' @param tau_values A numeric vector of time delay values.
#' @param emb_values A numeric vector of embedding dimension values.
#' @param vmin_values A numeric vector of minimum vertical line lengths.
#' @param threshold_values A numeric vector of threshold values for recurrence detection.
#' @param norm A character string specifying the distance norm to use. Possible values are:
#'   \itemize{
#'     \item `"euclidean"`: Euclidean distance.
#'     \item `"maximal"`: Maximum norm (Chebyshev distance).
#'     \item `"none"`: No normalization.
#'   }
#' @param calc_ENTR A logical value indicating whether to calculate entropy (`TRUE` or `FALSE`).
#' @param comp_platform A character string specifying the computing platform. Options may include
#'   `"cpu"`, `"nv_gpu"`, etc.
#'
#' @return A data frame with the following columns:
#'   \itemize{
#'     \item \code{LAM}: Laminarity percentage.
#'     \item \code{V}: Mean vertical line length.
#'     \item \code{Vmax}: Maximum vertical line length.
#'     \item \code{ENTR}: Entropy of the vertical line length distribution (if \code{calc_ENTR = TRUE}).
#'   }
#'
#' @details
#' Laminarity (\eqn{LAM}) is a measure in recurrence quantification analysis that describes the tendency of points
#' to form vertical lines in the recurrence plot. This function provides configurable parameters for
#' calculating LAM with options for normalization and entropy computation.
#'
#' @examples
#' # Example usage of accrqa_LAM
#' input <- c(1.0, 2.0, 3.0, 4.0)
#' tau_values <- c(1, 2)
#' emb_values <- c(2, 3)
#' vmin_values <- c(2, 3)
#' threshold_values <- c(0.1, 0.2)
#' norm <- "euclidean"
#' calc_ENTR <- TRUE
#' result <- accrqa_LAM(input, tau_values, emb_values, vmin_values, threshold_values, norm, calc_ENTR)
#'
#' @export
accrqa_LAM <- function(input, tau_values, emb_values, vmin_values, threshold_values, norm="euclidean", calc_ENTR=TRUE, platform)
{
  variables <- list(input = input, tau = tau_values, emb = emb_values, vmin = vmin_values, threshold = threshold_values)
  empty_vars <- names(variables)[sapply(variables, function(x) length(x) == 0)]
  
  if (length(empty_vars) > 0) {
    stop(paste("Number of delays, embedding, minimal vmin lengths or thresholds must be greater than zero or the input frame. The following are empy or null: ", paste(empty_vars, collapse = ", ")))
  }
  
  input_size <- length(input)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nVmins <- length(vmin_values)
  nThresholds <- length(threshold_values)
  output_size <- nTaus*nEmbs*nVmins*nThresholds*5
  
  if(input_size < 1) stop("n must be a positive integer!")
  
  norm_method <- switch_norm(norm)
  comp_platform <- switch_platform(platform)
  
  if (norm_method == 0) {
    stop("Normalization method to be used not recognized. Please use 'euclidean' or 'maximal'.")
  }
  
  if (comp_platform == 0) {
    stop("Platform to compute not recognized. Please use 'cpu' or 'nv_gpu'.")
  }
  
  if( any(variables$tau %% 1 != 0) == TRUE){
    warning("The delay values should be integers only, converting.")
  }
  
  if( any(variables$emb %% 1 != 0) == TRUE){
    warning("The tau values should be integers only, converting.")
  }
  
  if( any(variables$vmin %% 1 != 0) == TRUE){
    warning("The vmin values should be integers only, converting.")
  }
  
  rst <- .C("R_double_accrqa_LAM",
    output = double(length=output_size),
    input = as.double(input),
    input_size = as.integer(input_size),
    tau = as.integer(tau_values),
    tau_size = as.integer(nTaus),
    emb = as.integer(emb_values),
    emb_size = as.integer(nEmbs),
    vmin = as.integer(vmin_values),
    vmin_size = as.integer(nVmins),
    threshold = as.double(threshold_values),
    thr_size = as.integer(nThresholds),
    norm = as.integer(norm_method),
    entr = as.integer(calc_ENTR),
    platform = as.integer(comp_platform)
  )
  
  tidy_df <- expand.grid(
    Threshold = rst$threshold,
    Embedding = rst$emb, 
    Vmin = rst$vmin,
    Delay = rst$tau
  )
  tidy_df <- tidy_df[,c("Delay", "Embedding", "Vmin", "Threshold")]
  
  metrics <- as.data.frame(matrix(rst$output, ncol = 5, byrow = TRUE))
  colnames(metrics) <- c("LAM", "TT", "TTmax", "ENTR", "RR")
  result <- cbind(tidy_df, metrics)
  
  return(result)
}


#' Calculate Recurrence Rate (RR) in Recurrence Quantification Analysis (RQA)
#'
#' This function computes the recurrence rate (RR) for a given input time series based on the specified
#' delays, embedding dimensions, and thresholds. The function allows the user to specify normalization
#' and computational platform.
#'
#' @param input A numeric vector representing the input time series.
#' @param tau_values A numeric vector of time delay (\eqn{\tau}) values.
#' @param emb_values A numeric vector of embedding dimension (\eqn{m}) values.
#' @param threshold_values A numeric vector of threshold values for recurrence detection.
#' @param norm A character string specifying the normalization method. Defaults to `"euclidean"`. 
#'   Possible values are:
#'   \itemize{
#'     \item `"euclidean"`: Euclidean distance.
#'     \item `"maximal"`: Maximum norm (Chebyshev distance).
#'   }
#' @param platform A character string specifying the computational platform. Possible values are:
#'   \itemize{
#'     \item `"cpu"`: Use the CPU for computations.
#'     \item `"nv_gpu"`: Use an NVIDIA GPU for computations.
#'   }
#'
#' @return A data frame containing:
#'   \itemize{
#'     \item \code{Delay}: The delay (\eqn{\tau}) values used in the computation.
#'     \item \code{Embedding}: The embedding dimension (\eqn{m}) values used.
#'     \item \code{Threshold}: The threshold values used.
#'     \item \code{RR}: The computed recurrence rate for each combination of parameters.
#'   }
#'
#' @details
#' Recurrence rate (RR) quantifies the density of recurrence points in a recurrence plot.
#' This function uses a compiled C backend to efficiently compute RR based on the input parameters.
#' It performs validations on input lengths and ensures that parameters like delays and embeddings
#' are integers.
#'
#' @examples
#' # Example usage of accRQA_RR
#' input <- c(1.0, 2.0, 3.0, 4.0)
#' tau_values <- c(1, 2)
#' emb_values <- c(2, 3)
#' threshold_values <- c(0.1, 0.2)
#' norm <- "euclidean"
#' platform <- "cpu"
#' result <- accRQA_RR(input, tau_values, emb_values, threshold_values, norm, platform)
#' print(result)
#'
#' @export
accrqa_RR <- function(input, tau_values, emb_values, threshold_values, norm = "euclidean", platform)
{
  variables <- list(input = input, tau = tau_values, emb = emb_values, threshold = threshold_values)
  empty_vars <- names(variables)[sapply(variables, function(x) length(x) == 0)]
  
  if (length(empty_vars) > 0) {
    stop(paste("Number of delays, embedding or thresholds must be greater than zero or the input frame. The following are empy or null: ", paste(empty_vars, collapse = ", ")))
  }
  
  input_size <- length(input)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nThresholds <- length(threshold_values)
  output_size <- nTaus*nEmbs*nThresholds
  if(input_size < 1) stop("Length of the input time-series must be > 0!")
  
  norm_method <- switch_norm(norm)
  comp_platform <- switch_platform(platform)
  
  if (norm_method == 0) {
    stop("Normalization method to be used not recognized. Please use 'euclidean' or 'maximal'.")
  }
  
  if (comp_platform == 0) {
    stop("Platform to compute not recognized. Please use 'cpu' or 'nv_gpu'.")
  }

  if( any(variables$tau %% 1 != 0) == TRUE){
    warning("The tau values should be integers only, converting.")
  }
  
  if( any(variables$emb %% 1 != 0) == TRUE){
    warning("The emb values should be integers only, converting.")
  }
  
  rst <- .C("R_double_accrqa_RR",
      output = double(length=output_size),
      input = as.double(input),
      input_size = as.integer(input_size),
      tau = as.integer(tau_values),
      tau_size = as.integer(nTaus),
      emb = as.integer(emb_values),
      emb_size = as.integer(nEmbs),
      threshold = as.double(threshold_values),
      thr_size = as.integer(nThresholds),
      norm = as.integer(norm_method),
      platform = as.integer(comp_platform)
    )
    
    tidy_df <- expand.grid(
      Threshold = rst$threshold,
      Embedding = rst$emb, 
      Delay = rst$tau
    )
    tidy_df <- tidy_df[, c("Delay", "Embedding", "Threshold")]
    
    metrics <- as.data.frame(rst$output)
    colnames(metrics) <- c("RR")
    result <- cbind(tidy_df, metrics)
    
  return(result)
}
