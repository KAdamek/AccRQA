test_that("AccRQA results match -- RR",{
  input_data <- readRDS("testdata/input.rds")
  expected_RR <- readRDS("testdata/expected_RR.rds")
  
  
  emb <- c(3,4)
  tau <- c(6,10)
  threshold <- c(1.2, 1.8)
  
  results_RR <- accrqa_RR(input_data, 
                       tau_values = tau,
                       emb_values = emb,
                       threshold_values = threshold,
                       norm = "euclidean",
                       platform = "cpu")
  
  results_DET <- accrqa_DET(input_data, 
                          tau_values = tau,
                          emb_values = emb,
                          lmin_values = 2,
                          threshold_values = threshold,
                          norm = "euclidean",
                          calc_ENTR = TRUE,
                          platform = "cpu")
  
  results_DET <- subset(results_DET, select = -c(ENTR))
  expect_equal(results_RR, expected_RR, tolerance = 1e-4)
})

test_that("AccRQA results match -- DET",{
  input_data <- readRDS("testdata/input.rds")
  expected_DET <- readRDS("testdata/expected_DET.rds")
  
  emb <- c(3,4)
  tau <- c(6,10)
  threshold <- c(1.2, 1.8)
  
  results_DET <- accrqa_DET(input_data, 
                            tau_values = tau,
                            emb_values = emb,
                            lmin_values = 2,
                            threshold_values = threshold,
                            norm = "euclidean",
                            calc_ENTR = TRUE,
                            platform = "cpu")
  
  results_DET <- subset(results_DET, select = -c(ENTR))
  expect_equal(results_DET, expected_DET, tolerance = 1e-4)
  
  
})


test_that("AccRQA results match -- LAM",{
  input_data <- readRDS("testdata/input.rds")
  expected_LAM <- readRDS("testdata/expected_LAM.rds")
  
  emb <- c(3,4)
  tau <- c(6,10)
  threshold <- c(1.2, 1.8)
  
  results_LAM <- accrqa_LAM(input_data, 
                            tau_values = tau,
                            emb_values = emb,
                            vmin_values = 2,
                            threshold_values = threshold,
                            norm = "euclidean",
                            calc_ENTR = TRUE,
                            platform = "cpu")
  
  results_LAM <- subset(results_LAM, select = -c(ENTR))
  expect_equal(results_LAM, expected_LAM, tolerance = 1e-4)
  
})


# 
# expected <- data.frame(
#   Delay = c(6, 6, 6, 6, 10, 10, 10, 10),
#   Embedding = c(3, 3, 4, 4, 3, 3, 4, 4),
#   Vmin = c(2, 2, 2, 2, 2, 2, 2, 2),
#   Threshold = c(1.2, 1.8, 1.2, 1.8, 1.2, 1.8, 1.2, 1.8),
#   LAM = c(0.9958, 0.999, 0.9915, 0.99848, 0.99449, 0.99897, 0.99438, 0.99738),
#   TT   = c(6.87626, 10.1884, 5.77569, 8.30324, 6.33362, 9.15617, 5.3668, 7.53535),
#   TTmax = c(26, 40, 20, 26, 20, 31, 14, 20),
#   RR  = c(0.03407, 0.06586, 0.02182, 0.04217, 0.02452, 0.04794, 0.01641, 0.03136)
# )
