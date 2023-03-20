dyn.load("libAccRQA_R.so")

accrqa_DET <- function(input, tau_values, emb_values, lmin_values, threshold_values, norm, calc_ENTR)
{
  input_size <- length(input)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nLmins <- length(lmin_values)
  nThresholds <- length(thresholds)
  output_size <- nTaus*nEmbs*nLmins*nThresholds*5
  if(input_size < 1) stop("n must be a positive integer!")
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
    as.integer(norm),
    as.integer(calc_ENTR)
  )
  result <- data.frame(det = rst[["detp"]], l = rst[["lp"]], lmax = rst[["lmaxp"]], entr = rst[["entr"]])
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