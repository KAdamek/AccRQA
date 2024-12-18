#dyn.load("libAccRQA_R.so")
#' @useDynLib AccRQA, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

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
#' @return A list containing:
#'   - `DET`: The determinism values computed for the given input parameters.
#'   - `ENTR` (if `calc_ENTR = TRUE`): The entropy values corresponding to the DET computations.
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
accrqa_DET <- function(input, tau_values, emb_values, lmin_values, threshold_values, norm, calc_ENTR, platform)
{
  input_size <- length(input)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nLmins <- length(lmin_values)
  nThresholds <- length(threshold_values)
  output_size <- nTaus*nEmbs*nLmins*nThresholds*5
  if(input_size < 1) stop("n must be a positive integer!")
  
  norm_method <- switch(
    norm,
    "euclidean" = 1,
    "maximal" = 2,
    0  # Default case
  )
  
  comp_platform <- switch(
    platform,
    "cpu" = 1,
    "nv_gpu" = 1024,
    0
  )

  rst <- .C("R_double_accrqa_DET",
    output = double(length=output_size),
    as.double(input),
    as.integer(input_size),
    as.integer(tau_values),
    as.integer(nTaus),
    as.integer(emb_values),
    as.integer(nEmbs),
    as.integer(lmin_values),
    as.integer(nLmins),
    as.double(threshold_values),
    as.integer(nThresholds),
    as.integer(norm_method),
    as.integer(calc_ENTR),
    as.integer(comp_platform)
  )
  
  result <- data.frame(rst) #data.frame(det = rst[["detp"]], l = rst[["lp"]], lmax = rst[["lmaxp"]], entr = rst[["entr"]])
  return(result)
}


accrqa_LAM <- function(input, tau_values, emb_values, vmin_values, threshold_values, norm, calc_ENTR)
{
  input_size <- length(input)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nVmins <- length(vmin_values)
  nThresholds <- length(thresholds)
  output_size <- nTaus*nEmbs*nVmins*nThresholds*5
  if(input_size < 1) stop("n must be a positive integer!")
  rst <- .C("R_double_accrqa_LAM",
    output = double(length=output_size),
    as.double(input),
    as.integer(input_size),
    as.integer(tau_values),
    as.integer(nTaus),
    as.integer(emb_values),
    as.integer(nEmbs),
    as.integer(vmin_values),
    as.integer(nVmins),
    as.double(threshold_values),
    as.integer(nThresholds),
    as.integer(norm),
    as.integer(calc_ENTR)
  )
  result <- data.frame(det = rst[["detp"]], l = rst[["lp"]], lmax = rst[["lmaxp"]], entr = rst[["entr"]])
  return(result)
}


accRQA_RR <- function(input, tau_values, emb_values, threshold_values, norm)
{
  input_size <- length(input)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nThresholds <- length(thresholds)
  output_size <- nTaus*nEmbs*nThresholds
  if(input_size < 1) stop("Length of the input time-series must be > 0!")
  if(nThresholds < 1) stop("Number of thresholds must be > 0!")
  rst <- .C("R_double_accrqa_RR",
    output = double(length=output_size),
    as.double(input),
    as.integer(input_size),
    as.integer(tau_values),
    as.integer(nTaus),
    as.integer(emb_values),
    as.integer(nEmbs),
    as.double(threshold_values),
    as.integer(nThresholds),
    as.integer(norm)
  )
  result <- data.frame(RR = rst[["RRp"]])
  return(result)
}
