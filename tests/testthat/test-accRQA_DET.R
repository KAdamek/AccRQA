test_that("DET computation: function throws an error when input is NULL",{
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  lmin_values <- c(2,3,4)
  expect_error(accrqa_DET(NULL, 
                       tau_values, 
                       emb_values,
                       lmin_values,
                       threshold,
                       distance_type = "maximal",
                       comp_platform = "cpu"
                       ),
               regexp = ":  input"
)
})

test_that("DET computation: function throws an error when used not defined normalization method",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  lmin_values <- c(2,3,4)
  expect_error(accrqa_DET(sequence, 
                          tau_values, 
                          emb_values,
                          lmin_values,
                          threshold,
                          distance_type = "maxima",
                          comp_platform = "cpu"
  ),
  regexp = "Normalization method"
  )
})

test_that("DET computation: function throws an error when platform not recognized",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  lmin_values <- c(2,3,4)
  expect_error(accrqa_DET(sequence, 
                          tau_values, 
                          emb_values,
                          lmin_values,
                          threshold,
                          distance_type = "maximal",
                          comp_platform = "cp"
  ),
  regexp = "Platform to compute not recognized"
  )
})

test_that("DET computation: function throws an error when tau is NULL or negative",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- NULL
  emb_values <- c(1)
  lmin_values <- c(2,3,4)
  expect_error(accrqa_DET(sequence, 
                          tau_values, 
                          emb_values,
                          lmin_values,
                          threshold,
                          distance_type = "maximal",
                          comp_platform = "cpu"
                          ),
               regexp = ":  tau"
  )
  tau_values <- numeric(0)
  expect_error(accrqa_DET(sequence, 
                          tau_values, 
                          emb_values,
                          lmin_values,
                          threshold,
                          distance_type = "maximal",
                          comp_platform = "cpu"
  ),
  regexp = ":  tau"
  )
})

# Test that the DET function returns a data type list, and the result is data.frame
test_that("DET computation is correct for constant input",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  lmin_values <- c(2,3,4)
  result <- accrqa_DET(sequence, 
                       tau_values, 
                       emb_values,
                       lmin_values,
                       threshold,
                       distance_type = "maximal",
                       comp_platform = "cpu"
                       )
  
  expect_type(result, "list") # Fails if the function does not return a data.frame
  expect_s3_class(result, "data.frame") # Should pass for a data.frame
  
  tmp <-(1:lmin_values[which.max(lmin_values)-1])
  rst_vector <- rep(1.0 - 2*cumsum(tmp)/100, times = length(threshold)*length(tau_values))
  expect_true(identical(result$DET, rst_vector))
  expect_true(all(result$Lmax == length(sequence) - 1))#
  expect_true(all(result$RR == 1))#
  
})
  
