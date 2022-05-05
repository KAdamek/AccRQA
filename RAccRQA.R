dyn.load("libAccRQA.so")

accRQA_det <- function(input, threshold, taup, embp, lminp, norm)
{
  n <- length(input)
  device <- 0;
  if(n < 1) stop("n must be a positive integer!")
  rst <- .C("R_double_accrqaDeterminismGPU",
    detp  = double(length=1),
    lp    = double(length=1),
    lmaxp = double(length=1),
    as.double(input),
    as.integer(n),
    as.double(threshold),
    as.integer(taup),
    as.integer(embp),
    as.integer(lminp),
    as.integer(norm),
    as.integer(device)
  )
  result <- data.frame(det = rst[["detp"]], l = rst[["lp"]], lmax = rst[["lmaxp"]])
  return(result)
}

accRQA_lam <- function(input, threshold, taup, embp, lminp, norm)
{
  n <- length(input)
  device <- 0;
  if(n < 1) stop("n must be a positive integer!")
  rst <- .C("R_double_accrqaLaminarityGPU",
    lamp  = double(length=1),
    ttp    = double(length=1),
    ttmaxp = double(length=1),
    as.double(input),
    as.integer(n),
    as.double(threshold),
    as.integer(taup),
    as.integer(embp),
    as.integer(lminp),
    as.integer(norm),
    as.integer(device)
  )
  result <- data.frame(lam = rst[["lamp"]], tt = rst[["ttp"]], ttmax = rst[["ttmaxp"]])
  return(result)
}

accRQA_RR <- function(input, thresholds, taup, embp, norm)
{
  input_size <- length(input)
  nThresholds <- length(thresholds)
  device <- 0;
  if(input_size < 1) stop("Length of the input time-series must be > 0!")
  if(nThresholds < 1) stop("Number of thresholds must be > 0!")
  rst <- .C("R_double_accrqaRecurrentRateGPU",
    RRp  = double(length=nThresholds),
    as.double(input),
    as.integer(input_size),
    as.double(thresholds),
	as.integer(nThresholds),
    as.integer(taup),
    as.integer(embp),
    as.integer(norm),
    as.integer(device)
  )
  result <- data.frame(RR = rst[["RRp"]])
  return(result)
}