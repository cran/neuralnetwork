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
expect_error(
  nn_class_ind(character()),
  "`classes` must contain at least one value."
)
expect_error(
  nn_class_ind(c("a", NA)),
  "`classes` cannot contain missing values."
)

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
expect_error(
  nn_compute(fit_class),
  "`newdata` is required."
)

gw <- nn_generalized_weights(fit_class, iris[1:4, ], response = "setosa")
stopifnot(is.matrix(gw))
stopifnot(nrow(gw) == 4)
expect_error(
  nn_generalized_weights(fit_class, iris[1:4, ], epsilon = 0),
  "`epsilon` must be a positive finite number."
)

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
stopifnot(identical(fit_lbfgs$training_units, "function_evaluations"))
stopifnot(is.list(fit_lbfgs$convergence))
stopifnot(is.integer(fit_lbfgs$convergence$code))
stopifnot(length(fit_lbfgs$convergence$code) == 1L)
stopifnot("convergence" %in% names(summary(fit_lbfgs)))
stopifnot(any(grepl("function evaluation", capture.output(fit_lbfgs),
                    fixed = TRUE)))
stopifnot(any(grepl("Evaluations:", capture.output(summary(fit_lbfgs)),
                    fixed = TRUE)))

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
expect_error(
  nn_hessian(fit_tiny, tiny_data[1:3, ], y = tiny_data$Species[1:2]),
  "Outcome data must have one row per prediction row."
)
expect_error(
  nn_hessian(fit_tiny, tiny_data, epsilon = 0),
  "`epsilon` must be a positive finite number."
)
expect_error(
  nn_hessian(fit_tiny, tiny_data, max_params = 0),
  "`max_params` must be a positive integer."
)
expect_error(
  nn_confint(fit_tiny, tiny_data, level = 1),
  "`level` must be a number between 0 and 1."
)

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
expect_error(
  nn_evaluate(fit_reg, mtcars[1:3, ], y = mtcars$mpg[1:2]),
  "Outcome data must have one row per prediction row."
)

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
expect_error(
  nn_permutation_importance(fit_reg, mtcars[1:3, ], y = mtcars$mpg[1:2]),
  "Outcome data must have one row per prediction row."
)
expect_error(
  nn_permutation_importance(fit_reg),
  "`newdata` is required."
)
expect_error(
  nn_permutation_importance(fit_reg, mtcars, n_repeats = Inf),
  "`n_repeats` must be a positive integer."
)

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
expect_error(
  nn_fit(mpg ~ wt + hp, mtcars, hidden = 4, epochs = 1,
         callbacks = function(state) list(stop = "yes"),
         seed = 4, verbose = FALSE),
  "Callback `stop` values must be TRUE or FALSE."
)
expect_error(
  nn_fit(mpg ~ wt + hp, mtcars, hidden = 4, epochs = 1,
         callbacks = function(state) list(learning_rate = Inf),
         seed = 4, verbose = FALSE),
  "Callback `learning_rate` values must be positive finite scalars."
)

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
stopifnot(all(tuned$results$success))

tuned_continue <- nn_tune(
  Species ~ .,
  iris,
  grid = list(hidden = list(4, -1), learning_rate = c(0.01)),
  error_action = "continue",
  epochs = 2,
  validation_split = 0.2,
  seed = 15,
  verbose = FALSE
)
stopifnot(inherits(tuned_continue$best_model, "neuralnetwork"))
stopifnot(sum(!tuned_continue$results$success) == 1L)
stopifnot(any(grepl("`hidden` must contain positive integer layer sizes",
                    tuned_continue$results$error, fixed = TRUE),
              na.rm = TRUE))
expect_error(
  nn_tune(Species ~ ., iris, grid = list(hidden = list(-1)),
          error_action = "continue", epochs = 1, verbose = FALSE),
  "All tuning candidates failed or produced unavailable scores."
)

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
expect_error(
  nn_cv(Species ~ ., iris, repeats = Inf, epochs = 1, verbose = FALSE),
  "`repeats` must be a positive integer."
)

rare_cv <- data.frame(
  x = seq_len(5),
  y = factor(c("a", "a", "a", "b", "b"))
)
expect_error(
  nn_cv(y ~ x, rare_cv, k = 3, epochs = 1, verbose = FALSE),
  "`k` cannot exceed the smallest class size when `stratify = TRUE`."
)

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

bad_class_matrix <- nn_class_ind(iris$Species)
bad_class_matrix[1, ] <- 0
expect_error(
  nn_fit(as.matrix(iris[, 1:4]), bad_class_matrix, task = "classification",
         epochs = 1, verbose = FALSE),
  "Rows of a classification matrix must sum to 1."
)
bad_class_matrix <- nn_class_ind(iris$Species)
bad_class_matrix[1, 1] <- -0.1
bad_class_matrix[1, 2] <- 1.1
expect_error(
  nn_fit(as.matrix(iris[, 1:4]), bad_class_matrix, task = "classification",
         epochs = 1, verbose = FALSE),
  "Classification matrices must contain class indicators or probabilities in [0, 1]."
)

factor_data <- data.frame(
  y = factor(c("a", "b", "a", "b")),
  x = factor(c("u", "v", "u", "v"))
)
factor_fit <- nn_fit(y ~ x, factor_data, hidden = 2, epochs = 2,
                     seed = 16, verbose = FALSE)
expect_error(
  predict(factor_fit, data.frame(x = factor("w", levels = c("u", "v", "w")))),
  "`newdata` contains factor levels not seen during fitting."
)
expect_error(
  predict(factor_fit, data.frame(z = 1)),
  "Could not prepare `newdata` with the fitted predictors."
)

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
  nn_fit(Species ~ ., iris, hidden = "bad", verbose = FALSE),
  "`hidden` must contain positive integer layer sizes"
)
expect_error(
  nn_fit(Species ~ ., iris, hidden = 0, dropout = 0.1, verbose = FALSE),
  "`dropout` must be 0 when `hidden = 0`."
)
expect_error(
  nn_fit(Species ~ ., iris, dropout = NA_real_, verbose = FALSE),
  "`dropout` must be numeric values in [0, 1)."
)
expect_error(
  nn_fit(Species ~ ., iris, learning_rate = Inf, verbose = FALSE),
  "`learning_rate` must be a positive finite number."
)
expect_error(
  nn_fit(Species ~ ., iris, stepmax = Inf, verbose = FALSE),
  "`stepmax` must be a positive integer when supplied."
)
expect_error(
  nn_fit(Species ~ ., iris, beta1 = 1, verbose = FALSE),
  "`beta1` must be in [0, 1)."
)
expect_error(
  nn_fit(Species ~ ., iris, beta2 = NA_real_, verbose = FALSE),
  "`beta2` must be in [0, 1)."
)
expect_error(
  nn_fit(Species ~ ., iris, epsilon = 0, verbose = FALSE),
  "`epsilon` must be a positive finite number."
)
expect_error(
  nn_fit(mpg ~ wt + hp, mtcars, loss = "huber", huber_delta = Inf,
         verbose = FALSE),
  "`huber_delta` must be a positive finite number."
)
expect_error(
  nn_fit(Species ~ ., iris, patience = 0, verbose = FALSE),
  "`patience` must be a positive integer or Inf."
)
expect_error(
  nn_fit(Species ~ ., iris, min_delta = -1, verbose = FALSE),
  "`min_delta` must be a finite non-negative number."
)
expect_error(
  nn_fit(Species ~ ., iris, scale = c(TRUE, FALSE), verbose = FALSE),
  "`scale` must be TRUE or FALSE."
)
expect_error(
  nn_fit(Species ~ ., iris, seed = NA_real_, verbose = FALSE),
  "`seed` must be NULL or a non-negative integer."
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

stopifnot(identical(neuralnetwork:::.offset_seed(.Machine$integer.max, 0L),
                    as.integer(.Machine$integer.max)))
stopifnot(identical(neuralnetwork:::.offset_seed(.Machine$integer.max, 1L),
                    1L))

tmp_bad <- tempfile(fileext = ".rds")
saveRDS(list(not = "a model"), tmp_bad)
expect_error(
  nn_load(tmp_bad),
  "`path` does not contain a neuralnetwork model."
)
