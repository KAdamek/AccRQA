library(AccRQA)

input <- rnorm(1000, mean=70 , sd=10)
accrqa_DET(input, 0.5, 1, 1, 2, 2, FALSE, 1)

input_data <- matrix(runif(100), nrow = 10)
tau <- c(1, 2)
emb <- c(2, 3)
lmin <- 1
threshold <- 0.1
norm_method <- "euclidean"
calculate_entropy <- FALSE

accrqa_DET(
  input = input_data,
  tau_values = tau,
  emb_values = emb,
  lmin_values = lmin,
  threshold_values = threshold,
  2,
  calc_ENTR = calculate_entropy,
  1
)

input_data <- matrix(runif(100), nrow = 10)
tau <- c(1.0, 2)
emb <- c(2, 3, 4)
lmin <- 2
threshold <- 1
norm_method <- "euclidean"
calculate_entropy <- FALSE
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
results


input <- c(1.0, 2.0, 3.0, 4.0)
tau_values <- c(1, 2)
emb_values <- c(2, 3)
vmin_values <- c(2, 3)
threshold_values <- c(0.1, 0.2)
norm <- "euclidean"
calc_ENTR <- TRUE
result <- accrqa_LAM(input, tau_values, emb_values, vmin_values, threshold_values, norm, calc_ENTR, platform = comp_platform)
result


input <- c(1.0, 2.0, 3.0, 4.0)
tau_values <- c(1, 2)
emb_values <- c(2, 3)
threshold_values <- c(0.1, 0.2)
norm <- "euclidean"
platform <- "cpu"
result <- accRQA_RR(input, tau_values, emb_values, threshold_values, norm, platform)
print(result)


generate_recurrence_plot <- function(sequence, threshold, norm = "euclidean") {
  n <- length(sequence)
  rp <- matrix(0, n, n)
  for (i in 1:n) {
    for (j in 1:n) {
      distance <- if (norm == "euclidean") {
        abs(sequence[i] - sequence[j])
      } else {
        max(abs(sequence[i] - sequence[j]))
      }
      rp[i, j] <- as.integer(distance <= threshold)
    }
  }
  return(rp)
}

compute_rqa_metrics <- function(rp) {
  n <- nrow(rp)
  
  # Recurrence Rate (RR)
  RR <- sum(rp) / (n^2) * 100  # Percentage
  
  # Diagonal Lines for DET
  diag_lengths <- c()
  for (k in -n+1:n-1) {
    diag_line <- sum(rp[row(rp) - col(rp) == k])
    if (diag_line > 1) diag_lengths <- c(diag_lengths, diag_line)
  }
  DET <- sum(diag_lengths) / sum(rp) * 100  # Percentage
  
  # Vertical Lines for LAM
  vert_lengths <- c()
  for (j in 1:n) {
    vert_line <- sum(rp[, j])
    if (vert_line > 1) vert_lengths <- c(vert_lengths, vert_line)
  }
  LAM <- sum(vert_lengths) / sum(rp) * 100  # Percentage
  
  return(list(RR = RR, DET = DET, LAM = LAM))
}

test_rqa <- function(sequence, threshold, norm = "euclidean") {
  # Generate recurrence plot
  rp <- generate_recurrence_plot(sequence, threshold, norm)
  
  # Compute exact metrics
  exact_metrics <- compute_rqa_metrics(rp)
  
  # Run your RQA function
  rqa_result <- accrqa_LAM(
    input = sequence, 
    tau_values = 1, 
    emb_values = 1, 
    vmin_values = 1, 
    threshold_values = threshold, 
    norm = norm, 
    calc_ENTR = FALSE, 
    platform = "cpu"
  )
  
  # Compare results
  print("Exact Metrics:")
  print(exact_metrics)
  print("Computed Metrics:")
  print(rqa_result)
}

test_cases <- list(
  list(sequence = rep(1, 10), threshold = 0.1),          # Constant
  list(sequence = 1:10, threshold = 0.1),               # Linear
  list(sequence = rep(c(1, 2, 3), times = 3), threshold = 0.1), # Periodic
  list(sequence = runif(10), threshold = 0.1)           # Random
)

for (test in test_cases) {
  cat("Testing sequence:", test$sequence, "\n")
  test_rqa(test$sequence, test$threshold)
}

library(testthat)

test_that("RQA results match exact metrics", {
  sequence <- runif(10)
  threshold <- 0.1
  rp <- generate_recurrence_plot(sequence, threshold)
  exact <- compute_rqa_metrics(rp)
  computed <- accrqa_LAM(
    input = sequence, 
    tau_values = 1, 
    emb_values = 1, 
    vmin_values = 1, 
    threshold_values = threshold, 
    norm = "euclidean", 
    calc_ENTR = FALSE, 
    platform = "cpu"
  )
  expect_equal(computed$RR, exact$RR, tolerance = 1e-6)
  expect_equal(computed$DET, exact$DET, tolerance = 1e-6)
  expect_equal(computed$LAM, exact$LAM, tolerance = 1e-6)
})