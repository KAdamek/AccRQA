tolerance <- 1e-6

emb <- c(3,4)
tau <- c(6,10)
threshold <- c(1.2, 1.8)

test_that("AccRQA results match CRQA -- RR, euclidean",{
  input_data <- readRDS("testdata/input.rds")
  expected_RR <- readRDS("testdata/expected_CRQA_RR_euclidean.rds")
  
  results_RR <- accrqa_RR(input_data, 
                       tau_values = tau,
                       emb_values = emb,
                       threshold_values = threshold,
                       norm = "euclidean",
                       platform = "cpu")
  

  expect_equal(results_RR, expected_RR, tolerance = tolerance)
})

test_that("AccRQA results match CRQA -- RR, maximal",{
  input_data <- readRDS("testdata/input.rds")
  expected_RR <- readRDS("testdata/expected_CRQA_RR_maximal.rds")

    results_RR <- accrqa_RR(input_data, 
                          tau_values = tau,
                          emb_values = emb,
                          threshold_values = threshold,
                          norm = "maximal",
                          platform = "cpu")
  
  
  expect_equal(results_RR, expected_RR, tolerance = tolerance)
})

test_that("AccRQA results match CRQA -- DET, euclidean",{
  input_data <- readRDS("testdata/input.rds")
  expected_DET <- readRDS("testdata/expected_CRQA_DET_euclidean.rds")
  
  results_DET <- accrqa_DET(input_data, 
                            tau_values = tau,
                            emb_values = emb,
                            lmin_values = c(2),
                            threshold_values = threshold,
                            norm = "euclidean",
                            calc_ENTR = TRUE,
                            platform = "cpu")
  
  results_DET <- subset(results_DET, select = -c(Lmax))
  expect_equal(results_DET, expected_DET, tolerance = tolerance)
  
})

test_that("AccRQA results match CRQA -- DET, maximal",{
  input_data <- readRDS("testdata/input.rds")
  expected_DET <- readRDS("testdata/expected_CRQA_DET_maximal.rds")
  
  results_DET <- accrqa_DET(input_data, 
                            tau_values = tau,
                            emb_values = emb,
                            lmin_values = c(2),
                            threshold_values = threshold,
                            norm = "maximal",
                            calc_ENTR = TRUE,
                            platform = "cpu")
  
  results_DET <- subset(results_DET, select = -c(Lmax))
  expect_equal(results_DET, expected_DET, tolerance = tolerance)
  
})

test_that("AccRQA results match CRQA -- LAM, euclidean",{
  input_data <- readRDS("testdata/input.rds")
  expected_LAM <- readRDS("testdata/expected_CRQA_LAM_euclidean.rds")


  results_LAM <- accrqa_LAM(input_data, 
                            tau_values = tau,
                            emb_values = emb,
                            vmin_values = 2,
                            threshold_values = threshold,
                            norm = "euclidean",
                            calc_ENTR = TRUE,
                            platform = "cpu")
  
  results_LAM <- subset(results_LAM, select = -c(ENTR, TTmax))
  expect_equal(results_LAM, expected_LAM, tolerance = tolerance)
})


test_that("AccRQA results match CRQA -- LAM, maximal",{
  input_data <- readRDS("testdata/input.rds")
  expected_LAM <- readRDS("testdata/expected_CRQA_LAM_maximal.rds")
  
  results_LAM <- accrqa_LAM(input_data, 
                            tau_values = tau,
                            emb_values = emb,
                            vmin_values = 2,
                            threshold_values = threshold,
                            norm = "maximal",
                            calc_ENTR = TRUE,
                            platform = "cpu")
  
  results_LAM <- subset(results_LAM, select = -c(ENTR, TTmax))
  expect_equal(results_LAM, expected_LAM, tolerance = tolerance)
})


# #LAM
# expected <- data.frame(
#   Delay = c(6, 6, 6, 6, 10, 10, 10, 10),
#   Embedding = c(3, 3, 4, 4, 3, 3, 4, 4),
#   Vmin = c(2, 2, 2, 2, 2, 2, 2, 2),
#   Threshold = c(1.2, 1.8, 1.2, 1.8, 1.2, 1.8, 1.2, 1.8),
#   LAM = c(0.9957835, 0.9990018, 0.9915, 0.9984823, 0.9944929, 0.9989694, 0.9943792, 0.99738),
#   TT   = c(6.887626, 10.18866, 5.775685, 8.303236, 6.333618, 9.156171, 5.366795, 7.535354),
#   RR  = c(0.03407029, 0.06585667, 0.0218179, 0.04216551, 0.02452232, 0.04794121, 0.0164114, 0.03136195)
# )
# saveRDS(expected,"tests/testthat/testdata/expected_CRQA_LAM_euclidean.rds")
# 
# #LAM - maximal
# expected <- data.frame(
#   Delay = c(6, 6, 6, 6, 10, 10, 10, 10),
#   Embedding = c(3, 3, 4, 4, 3, 3, 4, 4),
#   Vmin = c(2, 2, 2, 2, 2, 2, 2, 2),
#   Threshold = c(1.2, 1.8, 1.2, 1.8, 1.2, 1.8, 1.2, 1.8),
#   LAM = c(0.9931003, 0.9973889, 0.9943935, 0.9972749, 0.992868, 0.9972379, 0.9916292, 0.9981024),
#   TT   = c(8.65891, 12.93356, 7.899189, 11.76216, 7.721594, 11.48852, 7.241674, 10.45196),
#   RR  = c(0.04987582, 0.0988475, 0.03775631, 0.07677615, 0.03567676, 0.07274186, 0.02795053, 0.05877578)
# )
# saveRDS(expected,"tests/testthat/testdata/expected_CRQA_LAM_maximal.rds")
# 
# #DET
# expected <- data.frame(
#   Delay = c(6, 6, 6, 6, 10, 10, 10, 10),
#   Embedding = c(3, 3, 4, 4, 3, 3, 4, 4),
#   Lmin = c(2, 2, 2, 2, 2, 2, 2, 2),
#   Threshold = c(1.2, 1.8, 1.2, 1.8, 1.2, 1.8, 1.2, 1.8),
#   DET = c(0.9988565, 0.9991866, 0.99917, 0.9992995, 0.9993956, 0.999416, 0.9996934, 0.99989),
#   L   = c(19.80633, 19.85697, 27.5337, 27.15812, 33.78207, 32.488, 57.0379, 47.2743),
#   ENTR = c(3.544372, 3.626616, 3.746358, 3.859276, 3.935241, 4.029226, 4.087776, 4.243792),
#   RR  = c(0.03407029, 0.06585667, 0.0218179, 0.04216551, 0.02452232, 0.04794121, 0.0164114, 0.03136195)
# )
# saveRDS(expected,"tests/testthat/testdata/expected_CRQA_DET_euclidean.rds")
# 
# #DET
# expected <- data.frame(
#   Delay = c(6, 6, 6, 6, 10, 10, 10, 10),
#   Embedding = c(3, 3, 4, 4, 3, 3, 4, 4),
#   Lmin = c(2, 2, 2, 2, 2, 2, 2, 2),
#   Threshold = c(1.2, 1.8, 1.2, 1.8, 1.2, 1.8, 1.2, 1.8),
#   DET = c(0.9923518, 0.9937105, 0.9953496, 0.9956827, 0.9966764, 0.996536, 0.9980198, 0.9976886),
#   L   = c(19.79292, 21.81006, 28.71724, 30.98503, 29.19675, 32.35281, 40.91513, 48.25811),
#   ENTR = c(3.44732, 3.560704, 3.727906, 3.83457, 3.680081, 3.826237, 3.755255, 4.157337),
#   RR  = c(0.04987582, 0.0988475, 0.03775631, 0.07677615, 0.03567676, 0.07274186, 0.02795053, 0.05877578)
# )
# saveRDS(expected,"tests/testthat/testdata/expected_CRQA_DET_maximal.rds")
# 
# #RR
# expected <- data.frame(
#   Delay = c(6, 6, 6, 6, 10, 10, 10, 10),
#   Embedding = c(3, 3, 4, 4, 3, 3, 4, 4),
#   Threshold = c(1.2, 1.8, 1.2, 1.8, 1.2, 1.8, 1.2, 1.8),
#   RR  = c(0.03407029, 0.06585667, 0.0218179, 0.04216551, 0.02452232, 0.04794121, 0.0164114, 0.03136195)
# )
# saveRDS(expected,"tests/testthat/testdata/expected_CRQA_RR_euclidean.rds")
# 
# #RR
# expected <- data.frame(
#   Delay = c(6, 6, 6, 6, 10, 10, 10, 10),
#   Embedding = c(3, 3, 4, 4, 3, 3, 4, 4),
#   Threshold = c(1.2, 1.8, 1.2, 1.8, 1.2, 1.8, 1.2, 1.8),
#   RR  = c(0.04987582, 0.0988475, 0.03775631, 0.07677615, 0.03567676, 0.07274186, 0.02795053, 0.05877578)
# )
# saveRDS(expected,"tests/testthat/testdata/expected_CRQA_RR_maximal.rds")
