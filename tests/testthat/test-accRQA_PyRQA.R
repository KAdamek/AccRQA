testthat::skip_on_cran()

tolerance <- 1e-3
emb <- c(3,4)
tau <- c(6,10)
threshold <- c(1.2, 1.8)

test_that("AccRQA results match PyRQA -- RR, euclidean",{
  input_data <- readRDS("testdata/input.rds")
  expected_RR <- readRDS("testdata/expected_PyRQA_RR_euclidean.rds")

  results_RR <- accrqa_RR(input_data, 
                       tau_values = tau,
                       emb_values = emb,
                       threshold_values = threshold,
                       distance_type = "euclidean",
                       comp_platform = "cpu")
  
  expect_equal(results_RR, expected_RR, tolerance = tolerance)
})

test_that("AccRQA results match PyRQA -- DET, euclidean",{
  input_data <- readRDS("testdata/input.rds")
  expected_DET <- readRDS("testdata/expected_PyRQA_DET_euclidean.rds")
  expected_DET <- subset(expected_DET, select = -c(L))

  results_DET <- accrqa_DET(input_data, 
                            tau_values = tau,
                            emb_values = emb,
                            lmin_values = c(2),
                            threshold_values = threshold,
                            distance_type = "euclidean",
                            calculate_ENTR = TRUE,
                            comp_platform = "cpu")
  
  results_DET <- subset(results_DET, select = -c(ENTR,L))
  expect_equal(results_DET, expected_DET, tolerance = tolerance)
})

test_that("AccRQA results match PyRQA -- LAM, euclidean",{
  input_data <- readRDS("testdata/input.rds")
  expected_LAM <- readRDS("testdata/expected_PyRQA_LAM_euclidean.rds")

  results_LAM <- accrqa_LAM(input_data, 
                            tau_values = tau,
                            emb_values = emb,
                            vmin_values = 2,
                            threshold_values = threshold,
                            distance_type = "euclidean",
                            calculate_ENTR = TRUE,
                            comp_platform = "cpu")
  
  results_LAM <- subset(results_LAM, select = -c(ENTR))
  expect_equal(results_LAM, expected_LAM, tolerance = tolerance)
})
