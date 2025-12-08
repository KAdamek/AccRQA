test_that("accrqa_RP basic call returns valid accrqa_rp object", {
  skip_if_not_installed("AccRQA")
  library(AccRQA)

  # Tiny, easy example
  input_data <- sin(seq(0, 2 * pi, length.out = 20))
  tau        <- 1
  emb        <- 2
  threshold  <- 0.1

  rp <- accrqa_RP(
    input_data    = input_data,
    tau           = tau,
    emb           = emb,
    threshold     = threshold,
    distance_type = "euclidean"
  )

  expect_s3_class(rp, "accrqa_rp")
  expect_true(inherits(rp, "accrqa"))

  expect_type(rp$output, "integer")
  expect_equal(rp$input_size, length(input_data))
  expect_equal(rp$tau, tau)
  expect_equal(rp$emb, emb)
  expect_equal(rp$threshold, threshold)
  expect_equal(rp$distance_type, "euclidean")

  # Length of output should be a perfect square
  expect_true(sqrt(length(rp$output)) == as.integer(sqrt(length(rp$output))))

  # rp_size should correspond to side length of the RP
  # (once you fix rp_size to 'corrected_size', this will pass)
  corrected_size <- rp$input_size - (rp$emb - 1L) * rp$tau
  expect_equal(corrected_size * corrected_size, length(rp$output))
})

test_that("accrqa_RP errors on empty or NULL arguments", {
  skip_if_not_installed("AccRQA")
  library(AccRQA)

  input_data <- sin(seq(0, 2 * pi, length.out = 20))

  expect_error(
    accrqa_RP(
      input_data = numeric(0),
      tau        = 1,
      emb        = 2,
      threshold  = 0.1,
      distance_type = "euclidean"
    ),
    "empty or NULL",
    fixed = FALSE
  )

  expect_error(
    accrqa_RP(
      input_data = input_data,
      tau        = numeric(0),
      emb        = 2,
      threshold  = 0.1,
      distance_type = "euclidean"
    ),
    "empty or NULL",
    fixed = FALSE
  )

  expect_error(
    accrqa_RP(
      input_data = input_data,
      tau        = 1,
      emb        = numeric(0),
      threshold  = 0.1,
      distance_type = "euclidean"
    ),
    "empty or NULL",
    fixed = FALSE
  )

  expect_error(
    accrqa_RP(
      input_data = input_data,
      tau        = 1,
      emb        = 2,
      threshold  = numeric(0),
      distance_type = "euclidean"
    ),
    "empty or NULL",
    fixed = FALSE
  )
})

test_that("accrqa_RP requires numeric input_data", {
  skip_if_not_installed("AccRQA")
  library(AccRQA)

  expect_error(
    accrqa_RP(
      input_data    = letters,
      tau           = 1,
      emb           = 2,
      threshold     = 0.1,
      distance_type = "euclidean"
    ),
    "`input_data` must be numeric"
  )
})

test_that("accrqa_RP enforces scalar tau, emb, and threshold", {
  skip_if_not_installed("AccRQA")
  library(AccRQA)

  input_data <- sin(seq(0, 2 * pi, length.out = 20))

  expect_error(
    accrqa_RP(
      input_data    = input_data,
      tau           = c(1, 2),
      emb           = 2,
      threshold     = 0.1,
      distance_type = "euclidean"
    ),
    "`tau` must be a scalar"
  )

  expect_error(
    accrqa_RP(
      input_data    = input_data,
      tau           = 1,
      emb           = c(2, 3),
      threshold     = 0.1,
      distance_type = "euclidean"
    ),
    "`emb` must be a scalar"
  )

  expect_error(
    accrqa_RP(
      input_data    = input_data,
      tau           = 1,
      emb           = 2,
      threshold     = c(0.1, 0.2),
      distance_type = "euclidean"
    ),
    "`threshold` must be a scalar"
  )
})

test_that("accrqa_RP checks distance_type correctly", {
  skip_if_not_installed("AccRQA")
  library(AccRQA)

  input_data <- sin(seq(0, 2 * pi, length.out = 20))

  # valid choices
  expect_silent(
    accrqa_RP(
      input_data    = input_data,
      tau           = 1,
      emb           = 2,
      threshold     = 0.1,
      distance_type = "euclidean"
    )
  )
  expect_silent(
    accrqa_RP(
      input_data    = input_data,
      tau           = 1,
      emb           = 2,
      threshold     = 0.1,
      distance_type = "maximal"
    )
  )

  # invalid
  expect_error(
    accrqa_RP(
      input_data    = input_data,
      tau           = 1,
      emb           = 2,
      threshold     = 0.1,
      distance_type = "l2"
    ),
    "'arg' should be one of \"euclidean\", \"maximal\""
  )
})

test_that("accrqa_RP integerises tau and emb with warnings when needed", {
  skip_if_not_installed("AccRQA")
  library(AccRQA)

  input_data <- sin(seq(0, 2 * pi, length.out = 30))

  expect_warning(
    rp <- accrqa_RP(
      input_data    = input_data,
      tau           = 1.5,
      emb           = 2,
      threshold     = 0.1,
      distance_type = "euclidean"
    ),
    "delay `tau` should be integer",
    fixed = FALSE
  )
  expect_true(is.integer(rp$tau))

  expect_warning(
    rp2 <- accrqa_RP(
      input_data    = input_data,
      tau           = 1,
      emb           = 2.7,
      threshold     = 0.1,
      distance_type = "euclidean"
    ),
    "embedding `emb` should be integer",
    fixed = FALSE
  )
  expect_true(is.integer(rp2$emb))
})

test_that("accrqa_RP checks corrected_size > 0", {
  skip_if_not_installed("AccRQA")
  library(AccRQA)

  input_data <- sin(seq(0, 2 * pi, length.out = 10))

  # emb and tau chosen so corrected_size <= 0
  expect_error(
    accrqa_RP(
      input_data    = input_data,
      tau           = 5,
      emb           = 3,
      threshold     = 0.1,
      distance_type = "euclidean"
    ),
    "must be positive; check 'tau' and 'emb'"
  )
})


#################### plot #########################
test_that("plot.accrqa_rp basic raster plot works and returns ggplot", {
  skip_if_not_installed("ggplot2")

  set.seed(1)
  x  <- seq(0, 2 * pi, length.out = 50)
  ts <- sin(x)

  # small RP object
  rp <- accrqa_RP(
    input_data    = ts,
    tau           = 1,
    emb           = 2,
    threshold     = 0.2,
    distance_type = "euclidean"
  )

  expect_s3_class(rp, "accrqa_rp")

  p <- plot(rp)  # default: style = "raster", summary = FALSE

  # patchwork objects are c("patchwork", "gg", "ggplot"),
  # but without summary we expect a plain ggplot
  expect_s3_class(p, "ggplot")
})

test_that("plot.accrqa_rp supports tile and point styles", {
  skip_if_not_installed("ggplot2")

  set.seed(2)
  ts <- sin(seq(0, 4 * pi, length.out = 60))

  rp <- accrqa_RP(
    input_data    = ts,
    tau           = 1,
    emb           = 2,
    threshold     = 0.15,
    distance_type = "maximal"
  )

  expect_silent(p_tile  <- plot(rp, style = "tile"))
  expect_silent(p_point <- plot(rp, style = "point"))

  expect_s3_class(p_tile,  "ggplot")
  expect_s3_class(p_point, "ggplot")
})

test_that("plot.accrqa_rp errors for invalid style", {
  skip_if_not_installed("ggplot2")

  ts <- sin(seq(0, 2 * pi, length.out = 40))
  rp <- accrqa_RP(
    input_data    = ts,
    tau           = 1,
    emb           = 2,
    threshold     = 0.1,
    distance_type = "euclidean"
  )

  expect_error(
    plot(rp, style = "invalid_style"),
    "'arg' should be one of "
  )
})

test_that("plot.accrqa_rp handles date axes correctly", {
  skip_if_not_installed("ggplot2")

  ts <- sin(seq(0, 2 * pi, length.out = 36))
  rp <- accrqa_RP(
    input_data    = ts,
    tau           = 1,
    emb           = 2,
    threshold     = 0.1,
    distance_type = "euclidean"
  )

  rp_len <- sqrt(length(rp$output))
  time_index <- as.Date("2020-01-01") + 0:(rp_len - 1)

  expect_silent(
    p <- plot(rp, x_dates = time_index, y_dates = time_index)
  )
  expect_s3_class(p, "ggplot")
})

test_that("plot.accrqa_rp validates x_dates and y_dates length", {
  skip_if_not_installed("ggplot2")

  ts <- sin(seq(0, 2 * pi, length.out = 36))
  rp <- accrqa_RP(
    input_data    = ts,
    tau           = 1,
    emb           = 2,
    threshold     = 0.1,
    distance_type = "euclidean"
  )

  # wrong length x_dates
  bad_dates <- as.Date("2020-01-01") + 0:4
  expect_error(
    plot(rp, x_dates = bad_dates),
    "Length of x_dates must be equal to RP size"
  )

  # wrong length y_dates
  rp_len <- sqrt(length(rp$output))
  x_ok   <- as.Date("2020-01-01") + 0:(rp_len - 1)
  y_bad  <- as.Date("2020-01-01") + 0:2

  expect_error(
    plot(rp, x_dates = x_ok, y_dates = y_bad),
    "Length of y_dates must be equal to RP size"
  )
})

test_that("plot.accrqa_rp summary panel works when patchwork is available", {
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("patchwork")

  # If accrqa_DET/LAM rely on GPU, you might want skip_on_cran() here:
  # skip_on_cran()

  ts <- sin(seq(0, 4 * pi, length.out = 80))
  rp <- accrqa_RP(
    input_data    = ts,
    tau           = 1,
    emb           = 2,
    threshold     = 0.15,
    distance_type = "euclidean"
  )

  # This calls accrqa_DET and accrqa_LAM inside plot.accrqa_rp
  expect_silent(
    p_sum <- plot(rp, summary = TRUE)
  )

  # patchwork returns class c("patchwork", "gg", "ggplot")
  expect_true("ggplot"   %in% class(p_sum))
  expect_true("patchwork" %in% class(p_sum))
})

