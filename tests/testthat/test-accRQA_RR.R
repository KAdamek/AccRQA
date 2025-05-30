test_that("RR computation: function throws an error when input is NULL",{
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  expect_error(accrqa_RR(NULL, 
                          tau_values, 
                          emb_values,
                          threshold,
                          norm = "maximal",
                          platform = "cpu"
  ),
  regexp = ":  input"
  )
})

test_that("RR computation: function throws an error when use of not defined norm",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  expect_error(accrqa_RR(sequence, 
                         tau_values, 
                         emb_values,
                         threshold,
                         norm = "maxima",
                         platform = "cpu"
  ),
  regexp = "Normalization method"
  )
})

test_that("RR computation: function throws an error when platform not recognized",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  expect_error(accrqa_RR(sequence, 
                         tau_values, 
                         emb_values,
                         threshold,
                         norm = "maximal",
                         platform = "cp"
  ),
  regexp = "Platform to compute not recognized"
  )
})

test_that("RR computation: function throws an error when tau is NULL or negative",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- NULL
  emb_values <- c(1)
  lmin_values <- c(2,3,4)
  expect_error(accrqa_RR(sequence, 
                          tau_values, 
                          emb_values,
                          threshold,
                          norm = "maximal",
                          platform = "cpu"
  ),
  regexp = ":  tau"
  )
  tau_values <- numeric(0)
  expect_error(accrqa_RR(sequence, 
                          tau_values, 
                          emb_values,
                          threshold,
                          norm = "maximal",
                          platform = "cpu"
  ),
  regexp = ":  tau"
  )
})

# Test that the RR function returns a data type list, and the result is data.frame
test_that("RR computation is correct for constant input",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  result <- accrqa_RR(sequence, 
                       tau_values, 
                       emb_values,
                       threshold,
                       norm = "maximal",
                       platform = "cpu"
  )
  
  expect_type(result, "list") # Fails if the function does not return a data.frame
  expect_s3_class(result, "data.frame") # Should pass for a data.frame
  
  expect_true(all(result$RR == 1))#
  
})
