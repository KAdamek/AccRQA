test_that("LAM computation: function throws an error when input is NULL",{
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  vmin_values <- c(2)
  expect_error(accrqa_LAM(NULL, 
                          tau_values, 
                          emb_values,
                          vmin_values,
                          threshold,
                          norm = "maximal",
                          TRUE,
                          platform = "cpu"
  ),
  regexp = ":  input"
  )
})

test_that("LAM computation: function throws an error when used not defined normalization method",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  vmin_values <- c(2)
  expect_error(accrqa_LAM(sequence, 
                          tau_values, 
                          emb_values,
                          vmin_values,
                          threshold,
                          norm = "maxima",
                          TRUE,
                          platform = "cpu"
  ),
  regexp = "Normalization method"
  )
})

test_that("LAM computation: function throws an error when used not known platform",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  vmin_values <- c(2)
  expect_error(accrqa_LAM(sequence, 
                          tau_values, 
                          emb_values,
                          vmin_values,
                          threshold,
                          norm = "maximal",
                          TRUE,
                          platform = "cp"
  ),
  regexp = "Platform to compute not recognized"
  )
})

test_that("LAM computation: function throws an error when tau is NULL or negative",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- NULL
  emb_values <- c(1)
  vmin_values <- c(2,3,4)
  expect_error(accrqa_LAM(sequence, 
                          tau_values, 
                          emb_values,
                          vmin_values,
                          threshold,
                          norm = "maximal",
                          TRUE,
                          platform = "cpu"
  ),
  regexp = ":  tau"
  )
  tau_values <- numeric(0)
  expect_error(accrqa_LAM(sequence, 
                          tau_values, 
                          emb_values,
                          vmin_values,
                          threshold,
                          norm = "maximal",
                          TRUE,
                          platform = "cpu"
  ),
  regexp = ":  tau"
  )
})

# Test that the LAM function returns a data type list, and the result is data.frame
test_that("LAM computation is correct for constant input",{
  sequence <- rep(1, 10)
  threshold <- c(0.1, 0.2)
  tau_values <- c(1,2)
  emb_values <- c(1)
  vmin_values <- c(2,3,4)
  result <- accrqa_LAM(sequence, 
                       tau_values, 
                       emb_values,
                       vmin_values,
                       threshold,
                       norm = "maximal",
                       TRUE,
                       platform = "cpu"
  )
  
  expect_type(result, "list") # Fails if the function does not return a data.frame
  expect_s3_class(result, "data.frame") # Should pass for a data.frame
  
  expect_true(all(result$LAM == 1))
  expect_true(all(result$RR == 1))#
  
})
