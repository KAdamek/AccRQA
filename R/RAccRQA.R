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


#' Find threshold that yields target RR (recurrence rate)
#'
#' @param input_data     Numeric vector (time series)
#' @param tau            Delay
#' @param emb            Embedding dimension
#' @param target_RR      Target recurrence(s) rate in \code{[0, 1]}. Can be scalar or vector.
#' @param epsilon        OPTIONAL Allowed error to find the desired RR. Default: 0.01
#' @param distance_type  OPTIONAL Distance type ("euclidean" or "maximal"). Default: "euclidean".
#' @param comp_platform  OPTIONAL Computer platform to use either "cpu" or "nv_gpu". Default: "cpu".
#' @param max_iter       OPTIONAL Maximum iterations to find the threshold before give up. Default: 20.
#' @param threshold_min  OPTIONAL Lower bound for search. Default: 0
#' @param threshold_max  OPTIONAL Upper bound for search (set automatically if NULL)
#' @param n_threshold   OPTIONAL How many thresholds per iteration (e.g. 10). Default: 10.
#' @param verbose        OPTIONAL If TRUE, prints progress. Default: FALSE.
#' @param ...     Further arguments (currently ignored but kept for S3 compatibility).
#' @importFrom utils tail
#'
#' @return Data frame with columns:
#'  \itemize{
#'    \item target_RR - target RR
#'    \item threshold - threshold to achieve desired RR
#'    \item RR_found  - RR at the found threshold
#'  }
#'
#' @examples
#' x <- runif(1000)
#' accrqa_RR_target(x, tau = 1, emb = 2, target_RR = 0.5, epsilon = 0.001)
#'
#' # multiple targets
#' accrqa_RR_target(x, tau = 1, emb = 2, target_RR = c(0.1,0.5,0.8), epsilon = 0.001)
#' @export
accrqa_RR_target <- function(
    input_data,
    tau,
    emb,
    target_RR,
    epsilon = 0.01,
    distance_type = "euclidean",
    comp_platform = "cpu",
    max_iter = 20,
    threshold_min = 0,
    threshold_max = NULL,
    n_threshold = 10,
    verbose = FALSE,
    ...){

	# input check
	if (!is.numeric(input_data)) stop("Time series must be numeric.")
	if (any(target_RR < 0 | target_RR > 1)) stop("Target RR must be in [0,1]")
	if (epsilon <= 0) stop("Epsilon must be positive.")
	if (n_threshold < 2) stop("`n_threshold` must be at least 2.")
	if (length(input_data) == 0) stop("Empty input data.")
	
	# upper bound auto
	if (is.null(threshold_max)){
		# rough estimate 
		threshold_max <- max(abs(input_data - mean(input_data)))*5
	}

	if (threshold_min > threshold_max) {
		stop("`threshold_min` must be <= `threshold_max`.", call. = FALSE)
	}

	rr_target_single <- function(target){

		thr_min <- threshold_min
		thr_max <- threshold_max
		best_thr <- NA_real_
		best_RR <- NA_real_

		for (iter in seq_len(max_iter)){
			thr_values <- seq(thr_min, thr_max, length.out = n_threshold)

			# compute RR
			rr_df <- accrqa_RR(input_data, tau, emb, thr_values, distance_type, comp_platform)
			rr_df <- rr_df[order(rr_df$Threshold), ]
			thr_sorted <- rr_df$Threshold
			RR_sorted <- rr_df$RR

			if (verbose){
				message(sprintf(
						"Iter %d: thr in [%.5g, %.5g], RR in [%.5g, %.5g]",
						iter, thr_sorted[1], utils::tail(thr_sorted, 1), RR_sorted[1], tail(RR_sorted, 1)
						)
				)
			}

			idx_best <- which.min(abs(RR_sorted - target))
			best_thr <- thr_sorted[idx_best]
			best_RR <- RR_sorted[idx_best]
			RR_min <- RR_sorted[1]
			RR_max <- RR_sorted[length(RR_sorted)]

				if (abs(best_RR - target) <= epsilon){
					if (verbose){
						message(sprintf("Converged at iter %d: threshold = %.5g, RR = %.5g.",
								iter, best_thr, best_RR
								)
						)
					}
					return(list(
						    threshold = best_thr, 
						    RR_found = best_RR
						    )
					)
				}

				if (target <= RR_min){
					if ((abs(best_RR - target) >= epsilon) || (isFALSE(all.equal(best_thr, 0)))){
						warning(sprintf("Cant find reasonable target_RR =%.2g. Please decrease the value threshold_min. Current threshold_min: %.5g", target, thr_min))
						return(
						       list(
							    threshold = NA,
							    RR_found = NA
						       )
						)
					}
					else {
						return(
						       list(
							    threshold = thr_sorted[1],
							    RR_found = RR_sorted[1]
						       )
						)
					}
				}

				if (target >= RR_max){
						if (abs(best_RR - target) >= epsilon){
							warning(sprintf("Cant find reasonable target_RR =%.2g. Please increase the value threshold_max. Current threshold_max: %.5g", target, thr_max))
							return(
							       list(
								    threshold = NA,
								    RR_found = NA
							       )
							)                                                                                                         				} 
						else{
							return(
							       list(
								    threshold = utils::tail(thr_sorted, 1),
								    RR_found = utils::tail(RR_sorted, 1)
							       )
							)
						}
					}
			

			j <- which(RR_sorted >= target)[1]
			if (j == 1){
				thr_min <- thr_sorted[j]
				thr_max <- thr_sorted[j + 1]
			} else {
				thr_min <- thr_sorted[j - 1]
				thr_max <- thr_sorted[j]
			}


#			# stop if interval is tiny
#			if (abs(thr_max - thr_min) < .Machine$double.eps^0.5){
#				if (verbose){
#					message("Threshold interval is below numeric tolerance")
#				}
#				break
#			}
		}

		if (verbose){
			message(sprintf("Max iterations reached. Best found: threshold = %.5g, RR = %.5g",
					best_thr, best_RR)
			)
		}

		list(threshold = best_thr, 
		     RR_found = best_RR)
	}

	# loop over targets, collect results
	res_list <- lapply(target_RR, rr_target_single)

	out <- data.frame(
	    		target_RR  = target_RR,
			threshold  = vapply(res_list, `[[`, numeric(1), "threshold"),
			RR_found   = vapply(res_list, `[[`, numeric(1), "RR_found"),
			row.names  = NULL
	)
	out
}

#' Compute a Recurrence Plot (RP)
#'
#' Calculates recurrence plot from supplied time-series and return the RP plot.
#'
#' @param input_data Numeric vector with the time series.
#' @param tau        Delay (integer, scalar).
#' @param emb        Embedding dimension (integer, scalar).
#' @param threshold  Threshold for recurrence (numeric, scalar).
#' @param distance_type Character string specifying distance:
#'                      one of "euclidean" or "maximal".
#'
#' @return An object of class "accrqa_rp" (and "accrqa") containing:
#'   \itemize{
#'     \item output       Integer vector of length \code{rp_size^2} with RP (0/1).
#'     \item input        Original input data (numeric vector).
#'     \item input_size   Length of the input series.
#'     \item tau          Delay (integer).
#'     \item emb          Embedding dimension (integer).
#'     \item threshold    Threshold used.
#'     \item distance_type Distance type as character.
#'     \item rp_size      Effective RP side length after embedding.
#'   }
#'
#' @details
#' The choice of \code{threshold} strongly controls the recurrence rate (RR).
#' Smaller thresholds produce sparser plots (low RR), larger thresholds denser
#' plots (high RR). Typical practice is to choose \code{threshold} such that
#' RR is in a reasonable range (e.g. 1-5-10%) and then compute DET, LAM, ENTR
#' on the resulting RP.
#'
#' The returned RP is an \eqn{N \times N} matrix, where \eqn{N} is the length
#' of the (possibly embedded) time series. It can be visualised with
#' \code{\link[graphics]{plot}()} or your own plotting routines.
#'
#' @examples
#' ts <- sin(2 * pi * (1:100) / 20)
#' rp <- accrqa_RP(ts, tau = 1, emb = 2, threshold = 0.5, distance_type = "euclidean")
#' plot(rp)
#' plot(rp, summary = TRUE)
#'
#' @export
accrqa_RP <- function(input_data, tau, emb, threshold, 
		      distance_type=c("euclidean", "maximal")
		      ){
	variables <- list(input_data = input_data, tau = tau, emb = emb, threshold = threshold)
	empty_vars <- names(variables)[vapply(variables, function(x) length(x) == 0L, logical(1))]
	if (length(empty_vars) > 0L) {
		stop(
		     "Number of delays, embedding, or thresholds must be greater than zero. ",
		     "The following arguments are empty or NULL: ",
		     paste(empty_vars, collapse = ", "),
		     call. = FALSE
		)
	}

	# check of input data to be numeric
	if (!is.numeric(input_data)) {
		stop("`input_data` must be numeric (vector).", call. = FALSE)
	}

	distance_type <- match.arg(distance_type)
	norm_method <- switch_norm(distance_type)
	if (norm_method == 0L){
		stop(
		     "Distance type not recognized. Please use 'euclidean' or 'maximal'.",
		     call. = FALSE
		)
	}

	# scalar tau/emb/threshold (current C backend expects scalars)
	if (length(tau) != 1L) {
		stop("`tau` must be a scalar. Vector delays are not yet supported in this function.", call. = FALSE)
	}
	if (length(emb) != 1L) {
		stop("`emb` must be a scalar. Vector embeddings are not yet supported in this function.", call. = FALSE)
	}
	if (length(threshold) != 1L) {
		stop("`threshold` must be a scalar in this function.", call. = FALSE)
	}

	# integer tau and emb
	if (tau %% 1 != 0) {
		warning("The delay `tau` should be integer; converting via `as.integer()`.")
		tau <- as.integer(tau)
	} else {
		tau <- as.integer(tau)
	}

	if (emb %% 1 != 0) {
		warning("The embedding `emb` should be integer; converting via `as.integer()`.")
		emb <- as.integer(emb)
	} else {
	emb <- as.integer(emb)
	}

	# setting the sizes
	input_size <- length(input_data)
	corrected_size <- input_size - (emb - 1)*tau
	if (corrected_size <= 0L){
		stop("'corrected_size' must be positive; check 'tau' and 'emb'", call. = FALSE)
	}

	output_size <- corrected_size*corrected_size

	# call C backend
	rst <- .C("R_double_accrqa_RP",
		output = integer(output_size),
		input = as.double(input_data),
		input_size = as.integer(input_size),
		tau = as.integer(tau),
		emb = as.integer(emb),
		threshold = as.double(threshold),
		distance_type = as.integer(norm_method)
	)

	# wrap into S3 obj
	obj <- list(
		    output = rst$output,
		    input = rst$input,
		    input_size = input_size,
		    tau = tau,
		    emb = emb,
		    threshold = as.numeric(threshold),
		    distance_type = distance_type,
		    rp_size = output_size
	)
	class(obj) <- c("accrqa_rp", "accrqa")
	obj
}

#' Plot method for AccRQA recurrence plot objects
#'
#' @param x       An object of class "accrqa_rp".
#' @param title   Plot title.
#' @param xlabel  X-axis label.
#' @param ylabel  Y-axis label.
#' @param style   One of "raster", "tile", or "point".
#' @param color_min Colour for 0 entries.
#' @param color_max Colour for 1 entries.
#' @param x_dates Optional vector of dates/times for the x-axis (length = RP size).
#' @param y_dates Optional vector of dates/times for the y-axis (length = RP size).
#' @param summary Optional boolean parameter for adding metadata to the right side of the plot.
#' @param ...     Further arguments (currently ignored but kept for S3 compatibility).
#' @importFrom patchwork wrap_plots plot_layout
#'
#' @examples
#' x <- seq(0, 10*pi, length.out = 200)
#' ts <- sin(x)
#' rp <- accrqa_RP(ts, tau = 1, emb = 2, threshold = 0.1, distance_type = "maximal")
#' # raster plot
#' plot(rp)
#' 
#' # tile style
#' plot(rp, style = "tile")
#'
#' # With dates:
#' time_index <- as.Date("2020-01-01") + 0:(sqrt(length(rp$output)) - 1)
#' plot(rp, x_dates = time_index, y_dates = time_index)
#'
#' plot(rp, x_dates = time_index, y_dates = time_index, summary = TRUE)
#' @export
plot.accrqa_rp <- function(x,
                           title   = "",
                           xlabel  = "Time",
                           ylabel  = "Time",
                           style   = c("raster", "tile", "point"),
                           color_min = "#eff3ff",
                           color_max = "#08519c",
                           x_dates = NULL,
                           y_dates = x_dates,
			   summary = FALSE,
                           ...) {

  style <- match.arg(style)

  # Infer RP size from output length
  n <- length(x$output)
  rp_size <- sqrt(n)
  if (rp_size != as.integer(rp_size)) {
    stop("accrqa_rp$output length is not a perfect square.")
  }
  rp_size <- as.integer(rp_size)

  rp_matrix <- matrix(x$output, nrow = rp_size, ncol = rp_size, byrow = TRUE)

  # Build data frame
  rp_df <- expand.grid(
    x = seq_len(rp_size),
    y = seq_len(rp_size)
  )
  rp_df$value <- factor(as.vector(rp_matrix), levels = c(0, 1))

  # Map indices to dates/times if provided
  if (!is.null(x_dates)) {
    if (length(x_dates) != rp_size) {
      stop("Length of x_dates must be equal to RP size.")
    }
    rp_df$x <- x_dates[rp_df$x]
  }

  if (!is.null(y_dates)) {
    if (length(y_dates) != rp_size) {
      stop("Length of y_dates must be equal to RP size.")
    }
    rp_df$y <- y_dates[rp_df$y]
  }

  # Base plot
  rp <- ggplot2::ggplot(rp_df, ggplot2::aes(x = rp_df$x, y = rp_df$y, fill = rp_df$value)) +
    ggplot2::coord_fixed(ratio = 1) +
    ggplot2::xlab(xlabel) +
    ggplot2::ylab(ylabel) +
    ggplot2::labs(title = title) +
    ggplot2::theme_classic() +
    ggplot2::theme(legend.position = "none")

  # If x/y are Date or POSIXct, the continuous scales will be overridden automatically.
  # (You can add special handling if you want.)

  if (style == "tile") {
    rp <- rp +
      ggplot2::geom_tile(width = 0.9, height = 0.9) +
      ggplot2::scale_fill_manual(values = c(`0` = color_min, `1` = color_max))
  } else if (style == "point") {
    rp <- rp +
      ggplot2::geom_point(ggplot2::aes(colour = rp_df$value), size = 0.5, stroke = 0, shape = 15) +
      ggplot2::scale_colour_manual(values = c(`0` = color_min, `1` = color_max)) +
      ggplot2::guides(fill = "none")
  } else if (style == "raster") {
    rp <- rp +
      ggplot2::geom_raster() +
      ggplot2::scale_fill_manual(values = c(`0` = color_min, `1` = color_max))
  }

  # if summary is false, just return the RP plot
  if (!isTRUE(summary)){
	  return(rp)
  }

    # else compute basic info
  det_df <- accrqa_DET(
                     input_data         = x$input,
                     tau_values         = x$tau,
                     emb_values         = x$emb,
                     threshold_values   = x$threshold,
                     distance_type      = x$distance_type,
		     lmin_values        = 2,
                     comp_platform      = "cpu"
  )
  lam_df <- accrqa_LAM(
                     input_data         = x$input,
                     tau_values         = x$tau,
                     emb_values         = x$emb,
                     threshold_values   = x$threshold,
                     distance_type      = x$distance_type,
		     vmin_values        = 2,
                     comp_platform      = "cpu"
  )
  info_df <- data.frame(
		label = c(
			"Input length",
			"Delay (tau)",
			"Embedding (emb)",
			"Threshold",
			"RR",
			"DET",
			"Distance"
		),
		value = c(
			  x$input_size,
			  x$tau,
			  x$emb,
			  format(x$threshold, digits = 5),
			  format(det_df$RR[1], digits = 5),
			  format(det_df$DET[1], digits = 5),
			  format(lam_df$LAM[1], digits = 5),
			  format(det_df$L, digits = 5),
			  det_df$Lmax,
			  format(lam_df$TT, digits = 5),
			  lam_df$TTmax,
			  format(det_df$ENTR[1], digits = 5),
			  format(lam_df$ENTR[1], digits = 5),
			  x$distance_type
		), 
		stringAsFactors = FALSE
  )

  info_text <- paste0(
		      "Summary\n",
		      "--------\n",
		      sprintf("Input length : %s\n", x$input_size),
		      sprintf("Delay (tau)  : %s\n", x$tau),
		      sprintf("Embedding    : %s\n", x$emb),
		      sprintf("Threshold    : %s\n", format(x$threshold, digits = 5)),
		      sprintf("RR           : %s\n", format(det_df$RR[1], digits = 5)),
		      sprintf("DET          : %s\n", format(det_df$DET[1], digits = 5)),
		      sprintf("LAM          : %s\n", format(lam_df$LAM[1], digits = 5)),
		      sprintf("L            : %s\n", format(det_df$L, digits = 5)),
		      sprintf("Lmax         : %s\n", det_df$Lmax),
		      sprintf("TT           : %s\n", format(lam_df$TT, digits = 5)),
		      sprintf("TTmax        : %s\n", lam_df$TTmax),
		      sprintf("ENTR (DET)   : %s\n", format(det_df$ENTR[1], digits = 5)),
		      sprintf("ENTR (LAM)   : %s\n", format(lam_df$ENTR[1], digits = 5)),
		      sprintf("Distance     : %s\n", x$distance_type)
  )  

  info_plot <- ggplot2::ggplot() + 
	  ggplot2::geom_text(
			     ggplot2::aes(x = 0, y = 0.5, label = info_text),
			     hjust = 0, vjust = 0.5,
			     family = "serif",
			     size = 4.2
	  ) + 
	  ggplot2::xlim(0, 1) + 
	  ggplot2::ylim(0, 1) +
	  ggplot2::theme_void()
		
	# ---- combine RP + summary using patchwork if available ----
	if (requireNamespace("patchwork", quietly = TRUE)) {
		combined <- rp + info_plot + patchwork::plot_layout(widths = c(2, 1))
		return(combined)
	} else {
		warning("Package 'patchwork' not installed; returning RP plot only.")
		return(rp)
	}
}

#' Calculate Determinism for Cross-Recurrence Quantification Analysis
#'
#' This function calculates the determinism (DET) for cross-recurrence quantification analysis (CRQA)
#' based on a set of input parameters, including time delay, embedding dimensions, minimum line length,
#' threshold values, and normalization.
#'
#' @param input_data A numeric matrix or data frame representing the input data for CRQA analysis.
#' @param tau_values A numeric vector specifying the time delay(s) to be used in the analysis.
#' @param emb_values A numeric vector specifying the embedding dimensions to be tested.
#' @param lmin_values A numeric vector specifying the minimum diagonal line lengths for DET computation.
#' @param threshold_values A numeric vector specifying the threshold values for recurrence computation.
#' @param distance_type A character string specifying the normalization method to be used. Options may include
#'   `"euclidean"`, `"maximal"`, etc.
#' @param calculate_ENTR A logical value indicating whether to calculate entropy (ENTR) along with DET.
#' @param comp_platform A character string specifying the computing platform. Options may include
#'   `"cpu"`, `"nv_gpu"`, etc.
#' @return A data frame containing:
#'   - `Delay`: Specific time delay from the values set in the parameters.
#'   - `Embedding`: Specific embedding dimension from the values set in the parameters.
#'   - `Lmin`: Minimal diagonal line lengths set for DET computation.
#'   - `DET`: The determinism values computed for the given input parameters.
#'   - `ENTR` (if `calculate_ENTR = TRUE`): The entropy values corresponding to the DET computations.
#'   - `RR`: RR values.
#'
#' @details
#' The function performs cross-recurrence quantification analysis (CRQA) using the specified parameters.
#' DET measures the proportion of recurrent points forming diagonal lines in the recurrence plot,
#' which indicates deterministic structures in the data. If `calculate_ENTR` is `TRUE`, the entropy of diagonal
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
#'   input_data = input_data,
#'   tau_values = tau,
#'   emb_values = emb,
#'   lmin_values = lmin,
#'   threshold_values = threshold,
#'   distance_type = norm_method,
#'   calculate_ENTR = calculate_entropy,
#'   comp_platform = comp_platform
#' )
#'
#' results
#' @export
accrqa_DET <- function(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type="euclidean", calculate_ENTR=TRUE, comp_platform)
{
  variables <- list(input_data = input_data, tau = tau_values, emb = emb_values, lmin = lmin_values, threshold = threshold_values)
  empty_vars <- names(variables)[sapply(variables, function(x) length(x) == 0)]
  
  input_size <- length(input_data)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nLmins <- length(lmin_values)
  nThresholds <- length(threshold_values)
  output_size <- nTaus*nEmbs*nLmins*nThresholds*5

  if (length(empty_vars) > 0) {
    stop(paste("Number of delays, embedding, minimal lengths or thresholds must be greater than zero or the input frame. The following are empty or null: ", paste(empty_vars, collapse = ", ")))
  }
  
  if( any(variables$tau %% 1 != 0) == TRUE){
    warning("The delay values should be integers only, converting.")
  }
  
  if( any(variables$emb %% 1 != 0) == TRUE){
    warning("The tau values should be integers only, converting.")
  }
  
  if(is.logical(calculate_ENTR) == FALSE){
    stop("Invalid value of calculate_ENTR. Should be TRUE or FALSE")
  }
  
  norm_method <- switch_norm(distance_type)
  comp_platform <- switch_platform(comp_platform)
  
  if (norm_method == 0) {
    stop("Normalization method to be used not recognized. Please use 'euclidean' or 'maximal'.")
  }
  
  if (comp_platform == 0) {
    stop("Platform to compute not recognized. Please use 'cpu' or 'nv_gpu'.")
  }
  
  rst <- .C("R_double_accrqa_DET",
    output = double(length=output_size),
    input = as.double(input_data),
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
    entr = as.integer(calculate_ENTR),
    comp_platform = as.integer(comp_platform)
  )
  
  tidy_df <- expand.grid(
    Lmin = rst$lmin,
    Threshold = rst$threshold,
    Embedding = rst$emb, 
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
#' @param input_data A numeric vector representing the input time series data.
#' @param tau_values A numeric vector of time delay values.
#' @param emb_values A numeric vector of embedding dimension values.
#' @param vmin_values A numeric vector of minimum vertical line lengths.
#' @param threshold_values A numeric vector of threshold values for recurrence detection.
#' @param distance_type A character string specifying the distance norm to use. Possible values are:
#'   \itemize{
#'     \item `"euclidean"`: Euclidean distance.
#'     \item `"maximal"`: Maximum norm (Chebyshev distance).
#'     \item `"none"`: No normalization.
#'   }
#' @param calculate_ENTR A logical value indicating whether to calculate entropy (`TRUE` or `FALSE`).
#' @param comp_platform A character string specifying the computing platform. Options may include
#'   `"cpu"`, `"nv_gpu"`, etc.
#'
#' @return A data frame with the following columns:
#'   \itemize{
#'     \item \code{LAM}: Laminarity percentage.
#'     \item \code{V}: Mean vertical line length.
#'     \item \code{Vmax}: Maximum vertical line length.
#'     \item \code{ENTR}: Entropy of the vertical line length distribution (if \code{calculate_ENTR = TRUE}).
#'   }
#'
#' @details
#' Laminarity (\eqn{LAM}) is a measure in recurrence quantification analysis that describes the tendency of points
#' to form vertical lines in the recurrence plot. This function provides configurable parameters for
#' calculating LAM with options for normalization and entropy computation.
#'
#' @examples
#' # Example usage of accrqa_LAM
#' input <- runif(100)
#' tau_values <- c(1, 2)
#' emb_values <- c(2, 3)
#' vmin_values <- c(2, 3)
#' threshold_values <- c(0.1, 0.2)
#' norm <- "euclidean"
#' calculate_ENTR <- TRUE
#' comp_platform <- "cpu"
#' result <- accrqa_LAM(
#'   input            = input,
#'   tau_values       = tau_values,
#'   emb_values       = emb_values,
#'   vmin_values      = vmin_values,
#'   threshold_values = threshold_values,
#'   distance_type    = norm,
#'   calculate_ENTR   = calculate_ENTR,
#'   comp_platform    = comp_platform
#' )
#'
#' result
#' @export
accrqa_LAM <- function(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type="euclidean", calculate_ENTR=TRUE, comp_platform)
{
  variables <- list(input_data = input_data, tau = tau_values, emb = emb_values, vmin = vmin_values, threshold = threshold_values)
  empty_vars <- names(variables)[sapply(variables, function(x) length(x) == 0)]
  
  if (length(empty_vars) > 0) {
    stop(paste("Number of delays, embedding, minimal vmin lengths or thresholds must be greater than zero or the input frame. The following are empy or null: ", paste(empty_vars, collapse = ", ")))
  }
  
  input_size <- length(input_data)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nVmins <- length(vmin_values)
  nThresholds <- length(threshold_values)
  output_size <- nTaus*nEmbs*nVmins*nThresholds*5
  
  if(input_size < 1) stop("n must be a positive integer!")
  
  norm_method <- switch_norm(distance_type)
  comp_platform <- switch_platform(comp_platform)
  
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
    input = as.double(input_data),
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
    entr = as.integer(calculate_ENTR),
    platform = as.integer(comp_platform)
  )
  
  tidy_df <- expand.grid(
    Vmin = rst$vmin,
    Threshold = rst$threshold,
    Embedding = rst$emb, 
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
#' @param input_data A numeric vector representing the input time series.
#' @param tau_values A numeric vector of time delay (\eqn{\tau}) values.
#' @param emb_values A numeric vector of embedding dimension (\eqn{m}) values.
#' @param threshold_values A numeric vector of threshold values for recurrence detection.
#' @param distance_type A character string specifying the normalization method. Defaults to `"euclidean"`. 
#'   Possible values are:
#'   \itemize{
#'     \item `"euclidean"`: Euclidean distance.
#'     \item `"maximal"`: Maximum norm (Chebyshev distance).
#'   }
#' @param comp_platform A character string specifying the computational platform. Possible values are:
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
#' # Example usage of accrqa_RR
#' input <- runif(100)
#' tau_values <- c(1, 2)
#' emb_values <- c(2, 3)
#' threshold_values <- c(0.1, 0.2)
#' norm <- "euclidean"
#' platform <- "cpu"
#' result <- accrqa_RR(input, tau_values, emb_values, threshold_values, norm, platform)
#' print(result)
#'
#' @export
accrqa_RR <- function(input_data, tau_values, emb_values, threshold_values, distance_type = "euclidean", comp_platform)
{
  variables <- list(input_data = input_data, tau = tau_values, emb = emb_values, threshold = threshold_values)
  empty_vars <- names(variables)[sapply(variables, function(x) length(x) == 0)]
  
  if (length(empty_vars) > 0) {
    stop(paste("Number of delays, embedding or thresholds must be greater than zero or the input frame. The following are empy or null: ", paste(empty_vars, collapse = ", ")))
  }
  
  input_size <- length(input_data)
  nTaus <- length(tau_values)
  nEmbs <- length(emb_values)
  nThresholds <- length(threshold_values)
  output_size <- nTaus*nEmbs*nThresholds

  if (input_size < 1L){
	  stop("Length of the input time-series must be > 0!")
  }
  
  norm_method <- switch_norm(distance_type)
  comp_platform <- switch_platform(comp_platform)
  
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
      input = as.double(input_data),
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
