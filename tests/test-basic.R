library(neuralnetwork)

expect_error <- function(expr, pattern) {
  err <- tryCatch(
    {
      suppressWarnings(force(expr))
      NULL
    },
    error = function(e) e
  )
  if (is.null(err)) {
    stop("Expected an error matching: ", pattern, call. = FALSE)
  }
  if (!grepl(pattern, conditionMessage(err), fixed = TRUE)) {
    stop(
      "Unexpected error message.\nExpected to contain: ", pattern,
      "\nActual: ", conditionMessage(err),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

fit_class <- nn_fit(
  Species ~ .,
  data = iris,
  hidden = c(8, 4),
  epochs = 6,
  batch_size = 32,
  learning_rate = 0.01,
  backend = "auto",
  validation_split = 0.1,
  seed = 1,
  verbose = FALSE
)

prob <- predict(fit_class, iris[1:4, ], type = "prob")
cls <- predict(fit_class, iris[1:4, ], type = "class")

stopifnot(inherits(fit_class, "neuralnetwork"))
stopifnot(is.matrix(prob))
stopifnot(nrow(prob) == 4)
stopifnot(ncol(prob) == 3)
stopifnot(all(abs(rowSums(prob) - 1) < 1e-8))
stopifnot(is.factor(cls))
stopifnot(identical(levels(cls), levels(iris$Species)))
stopifnot("train_metric" %in% names(fit_class$history))
stopifnot(identical(fit_class$metric_name, "accuracy"))
stopifnot(ncol(nn_class_ind(iris$Species[1:6])) == 3)
stopifnot(identical(nn_which_is_max(c(1, 3, 2)), 2L))

sigmoid_values <- neuralnetwork:::.stable_sigmoid(matrix(c(-1000, 0, 1000),
                                                         ncol = 1))
stopifnot(all(is.finite(sigmoid_values)))
stopifnot(sigmoid_values[1, 1] == 0)
stopifnot(sigmoid_values[2, 1] == 0.5)
stopifnot(sigmoid_values[3, 1] == 1)

set.seed(42)
dense_x <- matrix(stats::rnorm(40), 8, 5)
dense_w <- matrix(stats::rnorm(15), 5, 3)
dense_b <- stats::rnorm(3)
stopifnot(isTRUE(all.equal(
  neuralnetwork:::.dense(dense_x, dense_w, dense_b, "r"),
  neuralnetwork:::.dense(dense_x, dense_w, dense_b, "rcpp"),
  tolerance = 1e-12
)))

eval_class <- nn_evaluate(fit_class, iris)
stopifnot("accuracy" %in% names(eval_class$metrics))
stopifnot(all(c("balanced_accuracy", "macro_f1", "log_loss") %in%
                names(eval_class$metrics)))
stopifnot(is.table(eval_class$confusion))

computed <- nn_compute(fit_class, iris[1:3, ])
stopifnot(is.list(computed$neurons))
stopifnot(is.matrix(computed$net.result))

gw <- nn_generalized_weights(fit_class, iris[1:4, ], response = "setosa")
stopifnot(is.matrix(gw))
stopifnot(nrow(gw) == 4)

fit_multinom <- nn_multinom(
  Species ~ .,
  iris,
  epochs = 3,
  batch_size = 32,
  seed = 6,
  verbose = FALSE
)
stopifnot(length(fit_multinom$hidden) == 0)
stopifnot(ncol(predict(fit_multinom, iris[1:3, ], type = "prob")) == 3)
stopifnot(is.list(coef(fit_multinom)))
stopifnot(inherits(summary(fit_multinom), "summary.neuralnetwork"))

fit_lbfgs <- nn_fit(
  Species ~ .,
  iris,
  hidden = 4,
  optimizer = "lbfgs",
  epochs = 5,
  seed = 8,
  verbose = FALSE
)
stopifnot(identical(fit_lbfgs$optimizer, "lbfgs"))
stopifnot(is.factor(predict(fit_lbfgs, iris[1:3, ], type = "class")))

fit_momentum <- nn_fit(
  Species ~ .,
  iris,
  hidden = 4,
  optimizer = "momentum",
  epochs = 3,
  learning_rate = 0.01,
  momentum = 0.8,
  seed = 9,
  verbose = FALSE
)
stopifnot(identical(fit_momentum$optimizer, "momentum"))
stopifnot(is.factor(predict(fit_momentum, iris[1:3, ], type = "class")))

iris2 <- subset(iris, Species != "virginica")
tiny_data <- iris2[c(1:4, 51:54), ]
fit_tiny <- nn_fit(
  Species ~ Petal.Length,
  data = tiny_data,
  hidden = 0,
  optimizer = "grprop",
  epochs = 2,
  seed = 7,
  verbose = FALSE
)
h <- nn_hessian(fit_tiny, tiny_data)
ci <- nn_confint(fit_tiny, tiny_data)
stopifnot(is.matrix(h))
stopifnot(nrow(ci) == nrow(h))

binary_weights <- ifelse(iris2$Species == "versicolor", 2, 1)
fit_binary <- nn_fit(
  Species ~ .,
  iris2,
  hidden = c(6, 3),
  epochs = 4,
  batch_size = 16,
  learning_rate = 0.01,
  sample_weight = binary_weights,
  class_weight = "balanced",
  gradient_clip = 5,
  learning_rate_decay = 0.01,
  seed = 3,
  verbose = FALSE
)

binary_prob <- predict(fit_binary, iris2[1:5, ], type = "prob")
stopifnot(identical(fit_binary$task, "binary_classification"))
stopifnot(ncol(binary_prob) == 2)
stopifnot(all(abs(rowSums(binary_prob) - 1) < 1e-8))
stopifnot(all(fit_binary$history$gradient_norm >= 0))
eval_binary <- nn_evaluate(fit_binary, iris2)
stopifnot(all(c("sensitivity", "specificity", "f1") %in%
                names(eval_binary$metrics)))

fit_reg <- nn_fit(
  mpg ~ wt + hp,
  data = mtcars,
  hidden = 6,
  epochs = 6,
  batch_size = 8,
  learning_rate = 0.01,
  validation_split = 0.1,
  y_scale = TRUE,
  seed = 2,
  verbose = FALSE
)

pred <- predict(fit_reg, mtcars[1:3, ])
stopifnot(is.numeric(pred))
stopifnot(length(pred) == 3)
stopifnot(all(is.finite(pred)))
stopifnot(isTRUE(fit_reg$target_scaler$enabled))
stopifnot(identical(fit_reg$metric_name, "rmse"))

eval_reg <- nn_evaluate(fit_reg, mtcars)
stopifnot(all(c("rmse", "mae", "rsq") %in% names(eval_reg$metrics)))

fit_huber <- nn_fit(
  mpg ~ wt + hp,
  data = mtcars,
  hidden = 4,
  loss = "huber",
  huber_delta = 0.75,
  epochs = 4,
  batch_size = 8,
  learning_rate = 0.01,
  validation_split = 0.1,
  seed = 13,
  verbose = FALSE
)
stopifnot(identical(fit_huber$loss, "huber"))
stopifnot(identical(fit_huber$huber_delta, 0.75))
stopifnot(all(is.finite(fit_huber$history$train_loss)))

fit_nesterov <- nn_fit(
  mpg ~ wt + hp,
  data = mtcars,
  hidden = 4,
  optimizer = "nesterov",
  stepmax = 5,
  learning_rate = 0.01,
  threshold = 1e9,
  seed = 10,
  verbose = FALSE
)
stopifnot(identical(fit_nesterov$optimizer, "nesterov"))
stopifnot(isTRUE(fit_nesterov$stopped_early))
stopifnot(identical(fit_nesterov$stop_reason, "threshold"))

imp <- nn_permutation_importance(fit_reg, mtcars, metric = "mae",
                                 n_repeats = 2, seed = 12)
stopifnot(inherits(imp, "neuralnetwork_importance"))
stopifnot(all(c("feature", "importance", "metric") %in% names(imp)))
stopifnot(nrow(imp) == 2)
stopifnot(all(imp$metric == "mae"))

epochs_seen <- 0L
fit_callback <- nn_fit(
  mpg ~ wt + hp,
  data = mtcars,
  hidden = 4,
  epochs = 10,
  callbacks = function(state) {
    epochs_seen <<- state$epoch
    list(stop = state$epoch >= 2)
  },
  seed = 4,
  verbose = FALSE
)
stopifnot(isTRUE(fit_callback$stopped_early))
stopifnot(identical(fit_callback$stop_reason, "callback"))
stopifnot(epochs_seen == 2L)

tmp <- tempfile(fileext = ".rds")
nn_save(fit_reg, tmp)
fit_loaded <- nn_load(tmp)
stopifnot(inherits(fit_loaded, "neuralnetwork"))
stopifnot(all.equal(predict(fit_reg, mtcars[1:3, ]),
                    predict(fit_loaded, mtcars[1:3, ])))

tuned <- nn_tune(
  Species ~ .,
  iris,
  grid = list(hidden = list(4, c(6, 3)), learning_rate = c(0.01)),
  metric = "balanced_accuracy",
  epochs = 3,
  validation_split = 0.2,
  seed = 5,
  verbose = FALSE
)
stopifnot(inherits(tuned, "neuralnetwork_tune"))
stopifnot(inherits(tuned$best_model, "neuralnetwork"))
stopifnot(nrow(tuned$results) == 2)
stopifnot(all(tuned$results$metric == "balanced_accuracy"))

cv <- nn_cv(
  Species ~ .,
  iris,
  k = 3,
  metric = "f1",
  epochs = 2,
  hidden = 4,
  seed = 11,
  verbose = FALSE
)
stopifnot(inherits(cv, "neuralnetwork_cv"))
stopifnot(nrow(cv$results) == 3)
stopifnot("macro_f1" %in% cv$results$metric)

matrix_fit <- nn_fit(
  as.matrix(iris[, 1:4]),
  iris$Species,
  hidden = 3,
  epochs = 2,
  seed = 14,
  verbose = FALSE
)
matrix_prob <- predict(matrix_fit, as.matrix(iris[1:3, 1:4]), type = "prob")
stopifnot(is.matrix(matrix_prob))
stopifnot(nrow(matrix_prob) == 3)

new_level <- iris[1:2, ]
new_level$Species <- factor(new_level$Species,
                            levels = c(levels(iris$Species), "new_class"))
new_level$Species[1] <- "new_class"
expect_error(
  nn_evaluate(fit_class, new_level),
  "Truth contains classes not seen during fitting."
)

bad_x <- iris
bad_x$Sepal.Length[1] <- NA
expect_error(
  nn_fit(Species ~ ., bad_x, epochs = 1, verbose = FALSE),
  "missing values"
)
expect_error(
  nn_fit(Species ~ ., iris, hidden = -1, verbose = FALSE),
  "`hidden` must contain positive integer layer sizes"
)
expect_error(
  nn_fit(Species ~ ., iris, hidden = 0, dropout = 0.1, verbose = FALSE),
  "`dropout` must be 0 when `hidden = 0`."
)
expect_error(
  nn_fit(Species ~ ., iris, epochs = 1, sample_weight = rep(1, 3),
         verbose = FALSE),
  "`sample_weight` must be a non-negative finite numeric vector"
)
expect_error(
  predict(fit_reg, mtcars[1:2, ], type = "prob"),
  "`type = \"class\"` and `type = \"prob\"` are only valid for classification."
)
expect_error(
  predict(matrix_fit, matrix(0, nrow = 2, ncol = 3)),
  "`newdata` has 3 columns; expected 4."
)
expect_error(
  nn_evaluate(fit_reg, mtcars[, c("wt", "hp")]),
  "Provide `y` or pass `newdata` with the original response column."
)
expect_error(
  nn_load(tempfile(fileext = ".rds")),
  "`path` does not exist:"
)
expect_error(
  nn_save(fit_reg, tempfile("missing", tmpdir = file.path(tempdir(), "nope"))),
  "The directory for `path` does not exist:"
)

tmp_bad <- tempfile(fileext = ".rds")
saveRDS(list(not = "a model"), tmp_bad)
expect_error(
  nn_load(tmp_bad),
  "`path` does not contain a neuralnetwork model."
)
