library(neuralnetwork)
internal <- function(name) getFromNamespace(name, "neuralnetwork")
near <- function(a, b, tol = 2e-5) {
  stopifnot(isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = tol)))
}
fails <- function(expr, pattern) {
  e <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(e, "error"), grepl(pattern, conditionMessage(e), fixed = TRUE))
}

# One-row updates must retain row weights, including weights from class balance.
for (backend in c("r", "rcpp")) {
  fit_one <- function(w) nn_fit(matrix(0, 2, 1), c(0, 10), hidden = 0,
    optimizer = "sgd", batch_size = 1, epochs = 1, learning_rate = 0.1,
    sample_weight = w, scale = FALSE, y_scale = FALSE, shuffle = FALSE,
    backend = backend, seed = 1, verbose = FALSE)
  near(fit_one(c(1, 9))$params$b[[1]], 1.8)
  near(fit_one(c(9, 1))$params$b[[1]], 0.2)
  balanced <- nn_fit(matrix(0, 4, 1), factor(c("a", "a", "a", "b")),
    hidden = 0, optimizer = "sgd", batch_size = 1, epochs = 1,
    learning_rate = 0.1, class_weight = "balanced", scale = FALSE,
    shuffle = FALSE, backend = backend, seed = 1, verbose = FALSE)
  bias <- 0
  for (i in 1:4) {
    probability <- 1 / (1 + exp(-bias))
    weight <- c(2/3, 2/3, 2/3, 2)[i]
    bias <- bias - 0.1 * weight * (probability - as.numeric(i == 4))
  }
  near(balanced$params$b[[1]], bias, 1e-12)
}

set.seed(781)
x <- matrix(c(-1.2, 0.3, 0.9, 1.4, -0.4, 0.6, 0.7, -1.3), 4, 2)
w <- c(0.2, 0.8, 3, 1.2)
scenarios <- list(
  list(task = "regression", loss = "squared_error", y = matrix(c(1, 0.3, -1, 2, -2, 1, 0.5, 0), 4)),
  list(task = "regression", loss = "huber", y = matrix(c(-3, 2, 0.8, -1), 4)),
  list(task = "binary_classification", loss = "squared_error", y = matrix(c(0, 1, 0.3, 0.8), 4)),
  list(task = "classification", loss = "squared_error", y = matrix(c(1, 0.2, 0, 0.3, 0, 0.3, 1, 0.3, 0, 0.5, 0, 0.4), 4)))
gradient_cases <- batch_cases <- 0L
for (backend in c("r", "rcpp")) for (activation in c("relu", "leaky_relu", "tanh", "sigmoid")) {
  for (scenario in scenarios) {
    params <- internal(".init_params")(c(2, 3, 2, ncol(scenario$y)), activation)
    params$b[[1]] <- c(0.43, -0.21, 0.73)
    params$b[[2]] <- c(0.38, -0.47)
    flat <- internal(".flatten_params")(params)
    objective <- function(theta) {
      p <- internal(".unflatten_params")(theta, params)
      f <- internal(".forward")(x, p, activation, scenario$task, 0, FALSE, backend)
      z <- f$caches[[length(f$caches)]]$Z
      error <- f$output - scenario$y
      row_loss <- if (scenario$task == "regression") {
        if (scenario$loss == "huber") rowSums(ifelse(abs(error) <= 0.7,
          error^2 / 2, 0.7 * (abs(error) - 0.35))) else rowSums(error^2) / 2
      } else if (scenario$task == "binary_classification") {
        pmax(z[, 1], 0) - scenario$y[, 1] * z[, 1] + log1p(exp(-abs(z[, 1])))
      } else {
        shifted <- z - apply(z, 1, max)
        log(rowSums(exp(shifted))) - rowSums(scenario$y * shifted)
      }
      sum(w * row_loss) / sum(w) + 0.07 * sum(unlist(p$W)^2) / (2 * sum(w))
    }
    numeric_gradient <- vapply(seq_along(flat$values), function(j) {
      plus <- minus <- flat$values
      plus[j] <- plus[j] + 1e-6
      minus[j] <- minus[j] - 1e-6
      (objective(plus) - objective(minus)) / 2e-6
    }, numeric(1))
    gradient_cases <- gradient_cases + 1L
    for (size in 1:4) {
      batches <- utils::combn(1:4, size, simplify = FALSE)
      gradients <- lapply(batches, function(rows) {
        forward <- internal(".forward")(x[rows, , drop = FALSE], params,
          activation, scenario$task, 0, FALSE, backend)
        internal(".flatten_grads")(internal(".backward")(
          forward, params, scenario$y[rows, , drop = FALSE], activation,
          scenario$task, 0.07, w[rows], scenario$loss, 0.7,
          regularization_denom = sum(w), training_n = 4))
      })
      near(rowMeans(do.call(cbind, gradients)), numeric_gradient)
      batch_cases <- batch_cases + 1L
    }
  }
}

# Zero-weight outliers must not move scalers, formula bases or fitted parameters.
d <- data.frame(x = seq(-2, 2, length.out = 8))
d$y <- 1 + 0.8 * d$x + 0.1 * sin(1:8)
padded <- rbind(d, data.frame(x = c(1e200, -1e200), y = c(1e200, -1e200)))
for (backend in c("r", "rcpp")) for (optimizer in c("sgd", "momentum", "nesterov", "adam", "rprop", "grprop", "lbfgs")) {
  fit <- function(data, weights = NULL) nn_fit(y ~ x, data, hidden = 0,
    optimizer = optimizer, backend = backend, epochs = 8, batch_size = 3,
    sample_weight = weights, seed = 4, verbose = FALSE)
  reference <- fit(d)
  omitted <- fit(padded, c(rep(1, 8), 0, 0))
  near(predict(reference, d), predict(omitted, d), 1e-10)
  near(reference$scaler$center, omitted$scaler$center)
  near(reference$target_scaler$scale, omitted$target_scaler$scale)
  stopifnot(omitted$training_options$n == 8L,
            omitted$training_options$excluded_zero_weight_rows == 2L)
}
poly_padded <- rbind(d, data.frame(x = c(100, 120), y = c(10, 20)))
poly_reference <- nn_fit(y ~ poly(x, 2), d, hidden = 0, epochs = 30, seed = 5, verbose = FALSE)
poly_omitted <- nn_fit(y ~ poly(x, 2), poly_padded, sample_weight = c(rep(1, 8), 0, 0),
  hidden = 0, epochs = 30, seed = 5, verbose = FALSE)
near(predict(poly_reference, d), predict(poly_omitted, d), 1e-10)

for (optimizer in c("sgd", "momentum", "nesterov", "adam", "rprop", "grprop")) {
  fit <- function(weights) nn_fit(x, sin(x[, 1]), hidden = 3, optimizer = optimizer,
    sample_weight = weights, epochs = 5, batch_size = 3, seed = 2, verbose = FALSE)
  near(predict(fit(w), x), predict(fit(7 * w), x), 1e-9)
  params <- internal(".init_params")(c(2, 3, 1), "tanh")
  grad <- list(W = lapply(params$W, function(a) a * 0 + 0.3),
                b = lapply(params$b, function(a) a * 0 + 0.3))
  full <- internal(".init_optimizer_state")(params)
  small <- internal(".init_optimizer_state")(params, optimizer)
  step <- function(state) internal(".optimizer_step")(params, grad, state,
    optimizer, 0.02, 0.9, 0.999, 1e-8, 1, 0.8)$params
  near(unlist(step(full)), unlist(step(small)), 1e-12)
  stopifnot(as.numeric(object.size(small)) < as.numeric(object.size(full)))
}
stopifnot(length(internal(".init_optimizer_state")(params, "lbfgs")) == 0L)

# RPROP adapts to signs even when multiplying the gradients would underflow.
for (magnitude in c(1e-300, 1e-200, 1, 1e200, 1e300)) {
  same <- internal(".rprop_step")(1, magnitude, magnitude, 0.1)
  reversed <- internal(".rprop_step")(1, magnitude, -magnitude, 0.1)
  near(same$step, 0.12, 1e-12)
  near(same$param, 0.88, 1e-12)
  near(reversed$step, 0.05, 1e-12)
  near(reversed$param, 1, 1e-12)
  stopifnot(reversed$prev_grad == 0)
}
zero <- internal(".rprop_step")(1, 0, 1e300, 0.1)
near(zero$param, 1, 1e-12)
near(zero$step, 0.1, 1e-12)

# Near-hard soft targets and extreme regularization have finite exact answers.
almost_one <- 1 - .Machine$double.eps / 2
near(internal(".loss_from_output")(matrix(1, 1), matrix(almost_one, 1),
  "binary_classification", logits = matrix(1e18, 1)), (1 - almost_one) * 1e18, 1e-12)
near(internal(".loss_from_output")(matrix(0, 1), matrix(.Machine$double.eps / 2, 1),
  "binary_classification", logits = matrix(-1e18, 1)), .Machine$double.eps * 5e17, 1e-12)
near(internal(".loss_from_output")(matrix(c(1, 0), 1), matrix(c(1, 1e-300), 1),
  "classification", logits = matrix(c(1e308, -1e308), 1)), 2e8, 1e-12)
for (case in list(c(weight = 1e-200, l2 = 1e100, denom = 1e-300, expected = 1e200),
                  c(weight = 1e200, l2 = 1e-200, denom = 1e200, expected = 1e-200))) {
  p <- list(W = list(matrix(case[["weight"]], 1)), b = list(0))
  f <- internal(".forward")(matrix(0, 1), p, "tanh", "regression", 0, FALSE)
  g <- internal(".backward")(f, p, matrix(0, 1), "tanh", "regression", case[["l2"]],
    regularization_denom = case[["denom"]])
  near(g$W[[1]] / case[["expected"]], 1, 1e-12)
}
fails(nn_fit(x, x[, 1], optimizer = "lbfgs", gradient_clip = 0.1, verbose = FALSE), "gradient clipping")
clipped <- nn_fit(x, x[, 1], hidden = 0, gradient_clip = 0.1, epochs = 2, seed = 4, verbose = FALSE)
stopifnot(clipped$optimizer == "adam")
fails(nn_fit(x, x[, 1], hidden = 2^31, verbose = FALSE), "positive integer layer sizes")
fails(nn_fit(x + 1i, x[, 1], verbose = FALSE), "Complex inputs")
fails(nn_fit(x, x[, 1] + 1i, verbose = FALSE), "Complex inputs")
for (backend in c("r", "rcpp")) {
  tiny <- nn_fit(matrix(c(0, 1e-150), 2, 1), c(0, 1), hidden = 0,
    optimizer = "sgd", epochs = 1, seed = 1, backend = backend, verbose = FALSE)
  tiny$params$W[[1]][] <- 0
  fails(predict(tiny, matrix(1e308, 1)), "Predictor scaling exceeded")
}
fails(internal(".apply_target_scaler")(matrix(1e308, 1),
  list(enabled = TRUE, center = 0, scale = 1e-150)), "Target scaling exceeded")

# Exercise interacting fixes, not just each helper in isolation.
set.seed(284)
mixed <- data.frame(x = seq(-2, 2, length.out = 30),
                    group = factor(rep(c("a", "b", "c"), 10)))
mixed$y <- exp(0.8 + 0.3 * mixed$x + 0.1 * sin(1:30))
mixed$z <- 2 + mixed$x^2 + runif(30)
mixed_weights <- seq_len(30) / 10
mixed_weights[c(2, 8, 15, 27)] <- 0
for (backend in c("r", "rcpp")) {
  cv <- nn_cv(cbind(log_y = log(y), sqrt_z = sqrt(z)) ~ poly(x, 2) + group,
    data = mixed, k = 3, metric = "loss", sample_weight = mixed_weights,
    loss = "huber", huber_delta = 0.7, hidden = c(4, 2), optimizer = "adam",
    backend = backend, epochs = 4, batch_size = 2, seed = 18, verbose = FALSE)
  for (j in seq_along(cv$models)) {
    rows <- cv$folds[[j]]$test
    model <- cv$models[[j]]
    truth <- cbind(log_y = log(mixed$y[rows]), sqrt_z = sqrt(mixed$z[rows]))
    prediction <- predict(model, mixed[rows, ])
    weights <- mixed_weights[rows]
    error <- prediction - truth
    expected_rmse <- sqrt(sum(rowMeans(error^2) * weights) / sum(weights))
    scaled_error <- sweep(error, 2, model$target_scaler$scale, "/")
    row_loss <- rowSums(ifelse(abs(scaled_error) <= 0.7, scaled_error^2 / 2,
                              0.7 * (abs(scaled_error) - 0.35)))
    expected_loss <- sum(row_loss * weights) / sum(weights)
    near(cv$results$score[j], expected_loss, 1e-10)
    evaluation <- nn_evaluate(model, mixed[rows, ], sample_weight = weights)
    near(evaluation$metrics[["rmse"]], expected_rmse, 1e-10)
  }
}
xx <- matrix(rnorm(53 * 4), 53, 4, dimnames = list(NULL, paste0("v", 1:4)))
new_x <- matrix(rnorm(31 * 4), 31, 4, dimnames = list(NULL, colnames(xx)))
for (backend in c("r", "rcpp")) for (task in c("regression", "binary", "multiclass")) {
  yy <- switch(task, regression = cbind(a = xx[, 1], b = xx[, 2]^2),
               binary = factor(rep(c("no", "yes"), length.out = 53)),
               multiclass = factor(rep(letters[1:4], length.out = 53)))
  model <- nn_fit(xx, yy, hidden = c(5, 3), optimizer = "adam", backend = backend,
                  batch_size = 7, epochs = 3, seed = 14, verbose = FALSE)
  type <- if (task == "regression") "response" else "prob"
  expected <- predict(model, new_x, type = type)
  chunks <- split(1:31, rep(1:5, length.out = 31))
  actual <- matrix(NA_real_, nrow(expected), ncol(expected))
  for (rows in chunks) actual[rows, ] <- predict(model, new_x[rows, ], type = type)
  near(actual, expected, 1e-10)
  near(predict(model, new_x[, 4:1], type = type), expected, 1e-10)
  stopifnot(nrow(predict(model, new_x[0, ], type = type)) == 0L)
}

# Instrument actual training so the performance claim is not just a timing guess.
(function() {
  counter <- new.env(parent = emptyenv())
  counter$n <- 0L
  namespace <- asNamespace("neuralnetwork")
  trace(".forward", where = namespace,
    tracer = function() counter$n <- counter$n + 1L, print = FALSE)
  on.exit(untrace(".forward", where = namespace), add = TRUE)
  fit <- nn_fit(y ~ x, d, hidden = 3, optimizer = "lbfgs", epochs = 30, seed = 4, verbose = FALSE)
  stopifnot(fit$epochs > 5, counter$n <= fit$epochs + 5)
})()
cat("PASS:", gradient_cases, "deep-network gradient oracles;", batch_cases,
    "weighted batch averages; weight/scaler invariance, optimizer state, L-BFGS cache and extreme arithmetic.\n")
