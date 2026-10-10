library(neuralnetwork)
internal <- function(name) getFromNamespace(name, "neuralnetwork")
near <- function(a, b, tol = 1e-8) stopifnot(isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = tol)))
fails <- function(expr, pattern) {
  e <- tryCatch({ force(expr); NULL }, error = identity)
  if (!inherits(e, "error")) stop("Expected error: ", paste(deparse(substitute(expr)), collapse = " "))
  stopifnot(grepl(pattern, conditionMessage(e), fixed = TRUE))
}
x <- matrix(seq(-2, 2, length.out = 24), ncol = 1)
y <- sin(x[, 1]) + seq_len(24) / 20
w <- seq_len(24) / 10
tuned <- nn_tune(x, y, grid = list(hidden = list(0, 3), l2 = c(0, 0.1)),
                 validation_split = 0.25, metric = "loss", loss = "huber", seed = 91,
                 sample_weight = w, epochs = 15, optimizer = "adam", verbose = FALSE)
rows <- tuned$models[[1]]$validation$rows
stopifnot(all(vapply(tuned$models, function(m) identical(m$validation$rows, rows), logical(1))))
for (i in seq_along(tuned$models)) {
  m <- tuned$models[[i]]
  ev <- nn_evaluate(m, x[rows, , drop = FALSE], y[rows], sample_weight = w[rows])
  near(tuned$results$score[tuned$results$candidate_id == i], ev$loss)
  near(internal(".final_history")(m)$validation_loss, ev$loss)
}
fails(nn_tune(x, y, grid = list(seed = c(1, 2)), verbose = FALSE), "fixed across")
fails(nn_tune(x, y, maximize = 1, verbose = FALSE), "maximize")
fails(nn_tune(x, y, grid = list(loss = c("squared_error", "huber")), metric = "loss",
              hidden = 0, epochs = 3, seed = 2, verbose = FALSE), "same loss definition")
fails(nn_tune(x, y, grid = list(sample_weight = list(w, rev(w))), verbose = FALSE), "must be fixed")
train_tune <- nn_tune(x, y, grid = list(l2 = c(0, 10)), validation_split = 0,
                      hidden = 0, epochs = 3, metric = "loss", seed = 2, verbose = FALSE)
for (i in seq_along(train_tune$models)) {
  near(train_tune$results$score[train_tune$results$candidate_id == i],
       nn_evaluate(train_tune$models[[i]], x, y)$loss)
}

cv <- nn_cv(x, y, k = 3, repeats = 2, metric = "loss", sample_weight = w,
             loss = "huber", hidden = 0, optimizer = "adam", epochs = 8, seed = 52, verbose = FALSE)
for (i in seq_along(cv$models)) {
  rows <- cv$folds[[i]]$test
  near(cv$results$score[i], nn_evaluate(cv$models[[i]], x[rows, , drop = FALSE], y[rows],
                                      sample_weight = w[rows])$loss)
}
stopifnot(cv$summary$n_scored == 6L)
constant <- nn_cv(x, rep(1, 24), k = 3, metric = "rsq", hidden = 0, epochs = 5,
                  seed = 3, verbose = FALSE)
stopifnot(nrow(constant$results) == 3L, all(is.na(constant$results$score)),
          is.na(constant$summary$mean), constant$summary$n_scored == 0L,
          constant$summary$n_folds == 3L)
labels <- factor(rep(c("a", "b", "c"), 8), levels = c("a", "b", "c", "unused"))
folds <- nn_cv(x, labels, k = 3, hidden = 0, epochs = 5, seed = 62, verbose = FALSE)
for (f in folds$folds) stopifnot(all(table(droplevels(labels[f$test])) >= 2))
ind <- nn_class_ind(droplevels(labels))
folds <- nn_cv(x, ind, task = "classification", k = 3, hidden = 0, epochs = 5,
               seed = 62, verbose = FALSE)
for (f in folds$folds) stopifnot(all(colSums(ind[f$test, ]) >= 2))

single <- nn_fit(matrix(5, 1, 1), 3, hidden = 0, scale = FALSE, y_scale = FALSE,
                  optimizer = "sgd", epochs = 1, seed = 4, verbose = FALSE)
importance <- nn_permutation_importance(single, matrix(5, 1, 1), 3, n_repeats = 4, seed = 9)
near(importance$importance, 0)
empty <- nn_fit(y ~ 1, data.frame(y = y), hidden = 0, epochs = 20, seed = 4, verbose = FALSE)
stopifnot(nrow(nn_permutation_importance(empty, data.frame(y = y))) == 0L)
fails(nn_permutation_importance(empty, data.frame(y = rep(1, 24)), metric = "rsq"), "finite baseline")

for (backend in c("r", "rcpp")) for (task in c("regression", "classification", "binary")) {
  yy <- switch(task, regression = y, classification = droplevels(labels),
               binary = factor(rep(c("no", "yes"), 12)))
  for (optimizer in c("sgd", "momentum", "nesterov", "adam", "rprop", "grprop", "lbfgs")) {
    m <- nn_fit(x, yy, hidden = 3, optimizer = optimizer, backend = backend,
                epochs = 7, seed = 2, verbose = FALSE)
    stopifnot(all(is.finite(unlist(m$params))), all(is.finite(nn_evaluate(m, x, yy)$metrics)))
    near(internal(".final_history")(m)$train_metric,
         nn_evaluate(m, x, yy)$metrics[if (task == "regression") "rmse" else "accuracy"])
  }
}
zero <- nn_fit(x, y, hidden = 0, optimizer = "adam", epochs = 2, batch_size = 1,
               sample_weight = c(rep(0, 23), 1), shuffle = FALSE, seed = 12, verbose = FALSE)
stopifnot(all(is.finite(unlist(zero$params))))
encoded <- internal(".encode_y")(factor(c("a", "a", "a", "b", "b", "b")), "classification")
weighted <- internal(".make_sample_weights")(encoded, rep(1, 6), "balanced", 1:4)
near(weighted, c(2/3, 2/3, 2/3, 2, 1, 1))
balanced <- internal(".make_sample_weights")(
  internal(".encode_y")(factor(c("a", "b")), "classification"),
  c(1.3e308, 3e307), "balanced")
near(balanced / 8e307, c(1, 1))
best <- nn_fit(x, y, hidden = 0, optimizer = "sgd", epochs = 5, batch_size = 24,
               learning_rate = 0.05, min_delta = 100, seed = 2, verbose = FALSE)
stopifnot(best$best_epoch == which.min(best$history$train_loss), best$best_epoch > 1L)
auto <- nn_fit(x, y, hidden = 0, patience = 2, epochs = 5, seed = 2, verbose = FALSE)
stopifnot(auto$optimizer == "adam")
fails(nn_fit(x, y, optimizer = "lbfgs", patience = 2, verbose = FALSE), "does not support")
fails(nn_fit(x, y, optimizer = "sgd", learning_rate = 1e308, scale = FALSE,
             y_scale = FALSE, epochs = 5, verbose = FALSE), "non-finite")
fails(nn_fit(x, y, sample_weight = rep(1e308, 24), verbose = FALSE), "positive sum")
path <- tempfile(fileext = ".rds")
nn_save(best, path)
near(predict(nn_load(path), x), predict(best, x))
bad <- best
bad$params$W[[1]] <- matrix(NA_real_, 2, 2)
saveRDS(bad, path)
fails(nn_load(path), "invalid dimensions")
unlink(path)
fails(.Call("neuralnetwork_dense_bias", matrix(0, 2, 2), matrix(1, 1, 1), 0,
             PACKAGE = "neuralnetwork"), "dimension mismatch")
fails(.Call("neuralnetwork_dense_bias", matrix(0, 2, 1), matrix(1, 1, 1), numeric(),
             PACKAGE = "neuralnetwork"), "Bias length")
fails(.Call("neuralnetwork_softmax", matrix(0, 2, 0), PACKAGE = "neuralnetwork"), "at least one output")
fails(.Call("neuralnetwork_softmax", matrix(Inf, 2, 1), PACKAGE = "neuralnetwork"), "finite values")
stopifnot(identical(dim(.Call("neuralnetwork_softmax", matrix(0, 0, 2),
                             PACKAGE = "neuralnetwork")), c(0L, 2L)))
cat("PASS: common tuning holdout, weighted/undefined CV, single-row importance, 42 optimizer/task/backend fits and failure contracts.\n")
