#' Set number of threads used by AccRQA (OpenMP)
#' @param n integer >= 1
#' @return invisible(NULL)
#' @export
accrqa_set_num_threads <- function(n) {
	n <- as.integer(n)[1]
	if (is.na(n) || n < 1L){
	       	n <- 1L
	}
	.C("R_accrqa_set_num_threads", 
	   n, 
	   PACKAGE = "AccRQA"
	)
	invisible(NULL)
}

#' Get maximum available threads (OpenMP)
#' @return integer
#' @export
accrqa_get_max_threads <- function() {
	rst <- .C("R_accrqa_get_max_threads",
		 out = integer(1),
		 PACKAGE = "AccRQA"
	)
	rst$out[1]
}

#' Get current thread setting
#' @return integer
#' @export
accrqa_get_num_threads <- function() {
	rst <- .C("R_accrqa_get_num_threads",
		  out = integer(1),
		  PACKAGE = "AccRQA"
	)
	rst$out[1]
}

