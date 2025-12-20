library(AccRQA)
library(testthat)

# check_time_testthat.R
library(testthat)

test_dir <- "."
files <- list.files(test_dir, pattern="^test-.*\\.R$", full.names=TRUE)

total <- proc.time() * 0
results <- data.frame(file=character(), user=double(), system=double(), elapsed=double(),
                      status=character(), stringsAsFactors = FALSE)

for (f in files) {
  cat("\n--- Running", basename(f), "---\n")
  t0 <- proc.time()

  status <- "ok"
  tryCatch(
    test_file(f, reporter = "summary"),
    error = function(e) { status <<- paste0("error: ", conditionMessage(e)) },
    warning = function(w) { status <<- paste0("warning: ", conditionMessage(w)) }
  )

  dt <- proc.time() - t0
  print(dt)

  u <- unname(as.numeric(dt)[1])
  s <- unname(as.numeric(dt)[2])
  e <- unname(as.numeric(dt)[3])


  total <- total + dt
  results <- rbind(results, data.frame(
  	file    = basename(f),
	user    = u,
	system  = s,
	elapsed = e,
	status  = status,
	stringsAsFactors = FALSE
  ))
}

cat("\n=== TOTAL ===\n")
print(total)

cat("\n=== PER FILE ===\n")
print(results[order(-results$user), ], row.names = FALSE)

