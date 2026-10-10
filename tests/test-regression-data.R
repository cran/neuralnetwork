library(neuralnetwork)
internal <- function(name) getFromNamespace(name, "neuralnetwork")
near <- function(a, b, tol = 1e-9) stopifnot(isTRUE(all.equal(unname(a), unname(b), tolerance = tol)))
fails <- function(expr, pattern) {
  e <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(e, "error"), grepl(pattern, conditionMessage(e), fixed = TRUE))
}

# Independent weighted sums, including multiple columns with different offsets.
set.seed(816)
for (n in c(1, 2, 5, 9)) for (q in 1:4) for (rep in 1:5) {
  y <- matrix(rnorm(n * q), n, q) + rep(seq_len(q) * 100, each = n)
  pred <- y + matrix(rnorm(n * q), n, q)
  w <- runif(n)
  sse <- 0
  sst <- 0
  absolute <- 0
  for (j in seq_len(q)) {
    center <- sum(w * y[, j]) / sum(w)
    sse <- sse + sum(w * (pred[, j] - y[, j])^2)
    sst <- sst + sum(w * (y[, j] - center)^2)
    absolute <- absolute + sum(w * abs(pred[, j] - y[, j]))
  }
  expected <- c(rmse = sqrt(sse / sum(w) / q), mae = absolute / sum(w) / q,
                rsq = if (sst < 1e-20) NA_real_ else 1 - sse / sst)
  actual <- internal(".regression_metrics")(pred, y, w)
  near(actual, expected, 1e-8)
  shift <- rep(seq_len(q) * 700, each = n)
  near(actual, internal(".regression_metrics")(pred + shift, y + shift, w), 1e-8)
}

x <- matrix(seq(-2, 2, length.out = 20), ncol = 1, dimnames = list(NULL, "x"))
large_truth <- matrix(c(0, 1e200), 2, 1)
large <- internal(".regression_metrics")(large_truth + 1e199, large_truth)
near(large["rmse"] / 1e199, 1)
near(large["rsq"], 0.96)
near(internal(".regression_metrics")(matrix(0, 2, 1), matrix(c(-1e308, 1e308), 2, 1))["rsq"], 0)
near(internal(".metric_from_output")(large_truth + 1e199, large_truth, "regression") / 1e199, 1)
y <- cbind(a = x[, 1] + 1, b = 100 + 2 * x[, 1])
fit <- nn_fit(x, y, hidden = 0, optimizer = "adam", epochs = 5, seed = 41, verbose = FALSE)
ev <- nn_evaluate(fit, x, y)
near(internal(".final_history")(fit)$train_metric, ev$metrics["rmse"])
near(nn_evaluate(fit, x, y[, 2:1])$metrics, ev$metrics)
fails(nn_fit(x, c(y[, 1], 1), verbose = FALSE), "same number of rows")
fails(nn_fit(x[FALSE, , drop = FALSE], numeric(), verbose = FALSE), "at least one observation")
fails(nn_evaluate(fit, x[FALSE, , drop = FALSE], y[FALSE, ]), "at least one observation")
fails(nn_evaluate(fit, x, y[, 1]), "Outcome columns")
fails(predict(fit, matrix(1, 1, 1, dimnames = list(NULL, "wrong"))), "missing fitted predictor")
fails(nn_fit(cbind(x, x), y, verbose = FALSE), "unique")
fails(nn_fit(matrix("not numeric", 20, 1), y, verbose = FALSE), "numeric after encoding")
fails(nn_evaluate(fit, x, y, sample_weight = rep(0, 20)), "positive finite sum")
fails(nn_fit(matrix(c(-1e308, 1e308), 2, 1), 1:2, verbose = FALSE), "scaling exceeded")

dat <- data.frame(x = x[, 1], y = exp(x[, 1] + 3), z = x[, 1]^2)
logfit <- nn_fit(log(y) ~ x, dat, hidden = 0, epochs = 100, seed = 1, verbose = FALSE)
near(internal(".extract_truth")(logfit, dat, NULL), log(dat$y))
near(nn_evaluate(logfit, dat)$metrics, nn_evaluate(logfit, dat, log(dat$y))$metrics)
multi <- nn_fit(cbind(y, z) ~ x, dat, hidden = 0, epochs = 30, seed = 3, verbose = FALSE)
near(nn_evaluate(multi, dat)$metrics, nn_evaluate(multi, dat, cbind(y = dat$y, z = dat$z))$metrics)
polyfit <- nn_fit(y ~ poly(x, 2), dat, hidden = 0, epochs = 30, seed = 2, verbose = FALSE)
near(as.numeric(predict(polyfit, dat[1:3, ])), as.numeric(predict(polyfit, dat)[1:3]))
held <- c(1L, 3L, 5L)
polyheld <- nn_fit(y ~ poly(x, 2), dat, hidden = 0, validation_rows = held,
                   epochs = 30, seed = 2, verbose = FALSE)
train_basis <- poly(dat$x[-held], 2)
expected_basis <- predict(train_basis, dat$x[held])
actual_basis <- internal(".prepare_new_x")(dat[held, ], polyheld$blueprint)
near(as.numeric(actual_basis), as.numeric(expected_basis))
near(internal(".score_model_validation")(polyheld, "loss")$value,
     nn_evaluate(polyheld, dat[held, ])$loss)
dat$g <- factor(rep(c("a", "b", "c", "d"), 5))
old <- options(contrasts = c("contr.treatment", "contr.poly"))
catfit <- nn_fit(y ~ g, dat, hidden = 0, epochs = 20, seed = 3, verbose = FALSE)
df_fit <- nn_fit(dat["g"], dat$y, hidden = 0, epochs = 20, seed = 3, verbose = FALSE)
pred1 <- predict(catfit, dat)
pred2 <- predict(df_fit, dat["g"])
options(contrasts = c("contr.sum", "contr.poly"))
near(predict(catfit, dat), pred1)
near(predict(df_fit, dat["g"]), pred2)
options(old)
fails(nn_fit(y ~ x + offset(z), dat, verbose = FALSE), "offsets are not supported")
without_intercept <- nn_fit(y ~ x - 1, dat, hidden = 0, epochs = 20, seed = 3, verbose = FALSE)
stopifnot(length(without_intercept$params$b[[1]]) == 1L)

labels <- factor(rep(c("a", "b", "c"), length.out = 20))
ind <- nn_class_ind(labels)
clf <- nn_fit(x, ind, task = "classification", hidden = 0, epochs = 20, seed = 3, verbose = FALSE)
near(nn_evaluate(clf, x, labels)$metrics, nn_evaluate(clf, x, ind[, 3:1])$metrics)
soft <- ind * 0.7 + 0.1
soft_ev <- nn_evaluate(clf, x, soft)
near(soft_ev$metrics["log_loss"], -mean(rowSums(soft * log(pmax(soft_ev$probabilities, 1e-15)))))
bin <- nn_fit(x, matrix(rep(0:1, 10), ncol = 1), task = "classification", hidden = 0,
              epochs = 20, seed = 3, verbose = FALSE)
near(nn_evaluate(bin, x, matrix(rep(0:1, 10), ncol = 1))$metrics,
     nn_evaluate(bin, x, as.character(rep(0:1, 10)))$metrics)
cat("PASS: 80 weighted regression oracles; formula, basis, contrast and truth-encoding contracts.\n")
