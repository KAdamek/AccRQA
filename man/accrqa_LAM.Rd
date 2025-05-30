% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RAccRQA.R
\name{accrqa_LAM}
\alias{accrqa_LAM}
\title{Calculate Laminarity in Recurrence Quantification Analysis (RQA)}
\usage{
accrqa_LAM(
  input,
  tau_values,
  emb_values,
  vmin_values,
  threshold_values,
  norm = "euclidean",
  calc_ENTR = TRUE,
  platform
)
}
\arguments{
\item{input}{A numeric vector representing the input time series data.}

\item{tau_values}{A numeric vector of time delay values.}

\item{emb_values}{A numeric vector of embedding dimension values.}

\item{vmin_values}{A numeric vector of minimum vertical line lengths.}

\item{threshold_values}{A numeric vector of threshold values for recurrence detection.}

\item{norm}{A character string specifying the distance norm to use. Possible values are:
\itemize{
\item \code{"euclidean"}: Euclidean distance.
\item \code{"maximal"}: Maximum norm (Chebyshev distance).
\item \code{"none"}: No normalization.
}}

\item{calc_ENTR}{A logical value indicating whether to calculate entropy (\code{TRUE} or \code{FALSE}).}

\item{comp_platform}{A character string specifying the computing platform. Options may include
\code{"cpu"}, \code{"nv_gpu"}, etc.}
}
\value{
A data frame with the following columns:
\itemize{
\item \code{LAM}: Laminarity percentage.
\item \code{V}: Mean vertical line length.
\item \code{Vmax}: Maximum vertical line length.
\item \code{ENTR}: Entropy of the vertical line length distribution (if \code{calc_ENTR = TRUE}).
}
}
\description{
This function computes laminarity (LAM) based on the given input time series and RQA parameters.
}
\details{
Laminarity (\eqn{LAM}) is a measure in recurrence quantification analysis that describes the tendency of points
to form vertical lines in the recurrence plot. This function provides configurable parameters for
calculating LAM with options for normalization and entropy computation.
}
\examples{
# Example usage of accrqa_LAM
input <- c(1.0, 2.0, 3.0, 4.0)
tau_values <- c(1, 2)
emb_values <- c(2, 3)
vmin_values <- c(2, 3)
threshold_values <- c(0.1, 0.2)
norm <- "euclidean"
calc_ENTR <- TRUE
result <- accrqa_LAM(input, tau_values, emb_values, vmin_values, threshold_values, norm, calc_ENTR)

}
