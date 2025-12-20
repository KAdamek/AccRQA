test_that("accrqa_RR_target basic behaviour with valid input", {
  skip_if_not_installed("AccRQA") # or skip_on_cran() if this is the same package
  skip_on_cran()

  set.seed(1)
  ts <- sin(seq(0, 10, length.out = 20))

  res <- accrqa_RR_target(
    input_data    = ts,
    tau           = 1,
    emb           = 2,
    target_RR     = c(0.1, 0.5),
    distance_type = "euclidean",
    comp_platform = "cpu",
    threshold_min = 0,
    threshold_max = 1.5,
    n_thresholds  = 8,
    epsilon       = 0.01,
    max_iter      = 10,
    verbose       = FALSE
  )

  expect_s3_class(res, "data.frame")
  expect_equal(colnames(res),
               c("target_RR", "threshold", "RR_found"))
  expect_equal(nrow(res), 2L)
  expect_true(all(res$target_RR %in% c(0.1, 0.5)))
  expect_true(all(res$threshold >= 0))
  expect_true(all(res$threshold <= 1.5))
})

test_that("accrqa_RR_target errors on non-numeric input_data", {
  expect_error(
    accrqa_RR_target(
      input_data    = "not numeric",
      tau           = 1,
      emb           = 2,
      target_RR     = 0.1,
      threshold_min = 0,
      threshold_max = 1
    ),
    "Time series must be numeric."
  )
})

test_that("accrqa_RR_target errors when epsilon is negative", {
  expect_error(
    accrqa_RR_target(
      input_data    = runif(20),
      tau           = 1,
      emb           = 2,
      target_RR     = 0.1,
      threshold_min = 0,
      threshold_max = 1,
      epsilon       = -0.1
    ),
    "Epsilon must be positive."
  )
})

test_that("accrqa_RR_target errors when n_threshold is below 2", {
  expect_error(
    accrqa_RR_target(
      input_data    = runif(20),
      tau           = 1,
      emb           = 2,
      target_RR     = 0.1,
      threshold_min = 0,
      threshold_max = 1,
      n_threshold   = 1
    ),
    "`n_threshold` must be at least 2."
  )
})

test_that("accrqa_RR_target errors on empty input_data", {
  expect_error(
    accrqa_RR_target(
      input_data    = numeric(0),
      tau           = 1,
      emb           = 2,
      target_RR     = 0.1,
      threshold_min = 0,
      threshold_max = 1
    ),
    "Empty input data." 
  )
})

test_that("accrqa_RR_target errors for target_RR outside [0,1]", {
  set.seed(1)
  ts <- rnorm(100)

  expect_error(
    accrqa_RR_target(ts, 1, 2, target_RR = -0.1),
    "Target RR must be in \\[0,1\\]"
  )

  expect_error(
    accrqa_RR_target(ts, 1, 2, target_RR = 1.1),
    "Target RR must be in \\[0,1\\]"
  )
})

test_that("accrqa_RR_target errors when threshold_min > threshold_max", {
  set.seed(1)
  ts <- rnorm(20)

  expect_error(
    accrqa_RR_target(
      input_data    = ts,
      tau           = 1,
      emb           = 2,
      target_RR     = 0.2,
      threshold_min = 2,
      threshold_max = 1
    ),
    "`threshold_min` must be <= `threshold_max`."
  )
})

test_that("accrqa_RR_target warns and returns edge when target_RR is unreachable above range", {
  set.seed(1)
  ts <- sin(seq(0, 2 * pi, length.out = 50))

  # tiny threshold_max, so RR can’t reach 0.9
  expect_warning(
    res <- accrqa_RR_target(
      input_data    = ts,
      tau           = 1,
      emb           = 2,
      target_RR     = 0.9,
      threshold_min = 0,
      threshold_max = 0.01,
      n_thresholds  = 5,
      epsilon       = 0.01
    ),
    sprintf("Cant find reasonable target_RR =%.2g. Please increase the value threshold_max. Current threshold_max: %.5g", 0.9, 0.01)
  )

  expect_equal(nrow(res), 1L)
  expect_equal(res$target_RR, 0.9)
})

test_that("accrqa_RR_target warns and returns edge when target_RR is unreachable below range", {
  set.seed(1)
  ts <- sin(seq(0, 2 * pi, length.out = 50))

  # threshold_min very large → RR maybe always ~1, can't reach 0.01
  expect_warning(
    res <- accrqa_RR_target(
      input_data    = ts,
      tau           = 1,
      emb           = 2,
      target_RR     = 0.01,
      threshold_min = 1,
      threshold_max = 2,
      n_thresholds  = 5,
      epsilon       = 0.01
    ),
    sprintf("Cant find reasonable target_RR =%.2g. Please decrease the value threshold_min. Current threshold_min: %.5g", 0.01, 1)
  )

  expect_equal(nrow(res), 1L)
})

