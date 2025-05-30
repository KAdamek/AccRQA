% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RAccRQA.R
\name{accrqa_RR}
\alias{accrqa_RR}
\title{Calculate Recurrence Rate (RR) in Recurrence Quantification Analysis (RQA)}
\usage{
accrqa_RR(
  input,
  tau_values,
  emb_values,
  threshold_values,
  norm = "euclidean",
  platform
)
}
\arguments{
\item{input}{A numeric vector representing the input time series.}

\item{tau_values}{A numeric vector of time delay (\eqn{\tau}) values.}

\item{emb_values}{A numeric vector of embedding dimension (\eqn{m}) values.}

\item{threshold_values}{A numeric vector of threshold values for recurrence detection.}

\item{norm}{A character string specifying the normalization method. Defaults to \code{"euclidean"}.
Possible values are:
\itemize{
\item \code{"euclidean"}: Euclidean distance.
\item \code{"maximal"}: Maximum norm (Chebyshev distance).
}}

\item{platform}{A character string specifying the computational platform. Possible values are:
\itemize{
\item \code{"cpu"}: Use the CPU for computations.
\item \code{"nv_gpu"}: Use an NVIDIA GPU for computations.
}}
}
\value{
A data frame containing:
\itemize{
\item \code{Delay}: The delay (\eqn{\tau}) values used in the computation.
\item \code{Embedding}: The embedding dimension (\eqn{m}) values used.
\item \code{Threshold}: The threshold values used.
\item \code{RR}: The computed recurrence rate for each combination of parameters.
}
}
\description{
This function computes the recurrence rate (RR) for a given input time series based on the specified
delays, embedding dimensions, and thresholds. The function allows the user to specify normalization
and computational platform.
}
\details{
Recurrence rate (RR) quantifies the density of recurrence points in a recurrence plot.
This function uses a compiled C backend to efficiently compute RR based on the input parameters.
It performs validations on input lengths and ensures that parameters like delays and embeddings
are integers.
}
\examples{
# Example usage of accRQA_RR
input <- c(1.0, 2.0, 3.0, 4.0)
tau_values <- c(1, 2)
emb_values <- c(2, 3)
threshold_values <- c(0.1, 0.2)
norm <- "euclidean"
platform <- "cpu"
result <- accRQA_RR(input, tau_values, emb_values, threshold_values, norm, platform)
print(result)

}
