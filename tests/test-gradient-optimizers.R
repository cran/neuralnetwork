library(neuralnetwork)
get <- function(name) getFromNamespace(name, "neuralnetwork")
near <- function(a, b, tol = 2e-5) stopifnot(isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = tol)))
set.seed(819)
x <- matrix(c(-1.2, 0.3, 0.9, 1.4, -0.4, 0.6, 0.7, -1.3), 4, 2)
scenarios <- list(
  list(task = "regression", loss = "squared_error", y = matrix(c(1, 0.3, -1, 2, -2, 1, 0.5, 0), 4)),
  list(task = "regression", loss = "huber", y = matrix(c(-3, 2, 0.8, -1), 4)),
  list(task = "binary_classification", loss = "squared_error", y = matrix(c(0, 1, 0.3, 0.8), 4)),
  list(task = "classification", loss = "squared_error", y = matrix(c(1, 0.2, 0, 0.3, 0, 0.3, 1, 0.3, 0, 0.5, 0, 0.4), 4)))
count <- 0L
for (backend in c("r", "rcpp")) for (activation in c("relu", "leaky_relu", "tanh", "sigmoid")) {
  for (s in scenarios) for (dropout in c(0, 0.25)) {
    params <- get(".init_params")(c(2, 3, ncol(s$y)), activation)
    params$b[[1]] <- c(0.43, -0.21, 0.73)
    flat <- get(".flatten_params")(params)
    w <- c(0, 0.8, 2, 1.2)
    objective <- function(theta) {
      p <- get(".unflatten_params")(theta, params)
      set.seed(614)
      f <- get(".forward")(x, p, activation, s$task, dropout, TRUE, backend)
      z <- f$caches[[length(f$caches)]]$Z
      # Huber reference is evaluated directly, not through the gradient code.
      residual <- f$output - s$y
      rowloss <- if (s$task == "regression") {
        if (s$loss == "huber") rowSums(ifelse(abs(residual) <= 0.7, residual^2 / 2,
                                               0.7 * (abs(residual) - 0.35))) else rowSums(residual^2) / 2
      } else if (s$task == "binary_classification") {
        pmax(z[, 1], 0) - s$y[, 1] * z[, 1] + log1p(exp(-abs(z[, 1])))
      } else {
        maximum <- apply(z, 1, max)
        shifted <- z - maximum
        log(rowSums(exp(shifted))) - rowSums(s$y * shifted)
      }
      sum(w * rowloss) / sum(w) + 0.07 * sum(unlist(p$W)^2) / (2 * sum(w))
    }
    set.seed(614)
    f <- get(".forward")(x, params, activation, s$task, dropout, TRUE, backend)
    analytic <- get(".flatten_grads")(get(".backward")(f, params, s$y, activation,
                                         s$task, 0.07, w, s$loss, 0.7))
    numeric <- vapply(seq_along(flat$values), function(j) {
      plus <- minus <- flat$values
      plus[j] <- plus[j] + 1e-6
      minus[j] <- minus[j] - 1e-6
      (objective(plus) - objective(minus)) / 2e-6
    }, numeric(1))
    near(analytic, numeric)
    count <- count + 1L
  }
}

# Saturated logits must not make the objective flat while the gradient is nonzero.
for (backend in c("r", "rcpp")) {
  params <- list(W = list(matrix(0, 1, 1)), b = list(1000))
  xx <- matrix(0, 2, 1)
  yy <- matrix(0, 2, 1)
  near(get(".dataset_loss")(xx, yy, params, "tanh", "binary_classification", 0, backend), 1000)
  f <- get(".forward")(xx, params, "tanh", "binary_classification", 0, FALSE, backend)
  near(get(".backward")(f, params, yy, "tanh", "binary_classification", 0)$b[[1]], 1)
  multi <- list(W = list(matrix(0, 1, 3)), b = list(c(1000, 0, -1000)))
  yy <- matrix(c(0, 0, 0, 0, 1, 1), 2)
  near(get(".dataset_loss")(xx, yy, multi, "tanh", "classification", 0, backend), 2000)
  f <- get(".forward")(xx, multi, "tanh", "classification", 0, FALSE, backend)
  near(get(".backward")(f, multi, yy, "tanh", "classification", 0)$b[[1]], c(1, 0, -1))
}
huge <- list(W = list(matrix(c(1e300, -1e300), 2, 1)), b = list(1e300))
clipped <- get(".clip_grads")(huge, 2)
near(get(".gradient_norm")(clipped), 2)
params <- list(W = list(matrix(c(0.3, 0.5), 2, 1)), b = list(0.2))
f <- get(".forward")(x[1:2, ], params, "tanh", "regression", 0, FALSE)
g0 <- get(".backward")(f, params, matrix(1, 2, 1), "tanh", "regression", 0)
g1 <- get(".backward")(f, params, matrix(1, 2, 1), "tanh", "regression", 0.4,
                       regularization_denom = 100)
near(g1$W[[1]] - g0$W[[1]], 0.4 * params$W[[1]] / 100)
near(g1$b[[1]], g0$b[[1]])
large <- list(W = list(matrix(1e200, 1, 1)), b = list(0))
near(get(".dataset_loss")(matrix(0, 2, 1), matrix(0.5, 2, 1), large,
                           "tanh", "binary_classification", 0), log(2))
near(get(".l2_penalty")(large, 1e-200, NULL, 2) / 1e200, 0.25)
near(get(".dataset_loss")(matrix(0, 2, 1), matrix(0, 2, 1),
                           list(W = list(matrix(0, 1, 1)), b = list(1000)),
                           "tanh", "binary_classification", 0, weights = c(1e308, 0)), 1000)

# Scalar, independently updated state for every supported first-order rule.
for (optimizer in c("sgd", "momentum", "nesterov", "adam", "rprop", "grprop")) {
  params <- list(W = list(matrix(0.7, 1, 1)), b = list(0.7))
  state <- get(".init_optimizer_state")(params)
  value <- 0.7
  velocity <- m <- v <- previous <- 0
  delta <- 0.1
  for (step in 1:4) {
    g <- c(0.3, 0.5, -0.2, -0.4)[step]
    if (optimizer == "sgd") value <- value - 0.02 * g
    if (optimizer %in% c("momentum", "nesterov")) {
      old_velocity <- velocity
      velocity <- 0.8 * velocity - 0.02 * g
      value <- if (optimizer == "momentum") value + velocity else value - 0.8 * old_velocity + 1.8 * velocity
    }
    if (optimizer == "adam") {
      m <- 0.9 * m + 0.1 * g
      v <- 0.999 * v + 0.001 * g^2
      value <- value - 0.02 * (m / (1 - 0.9^step)) / (sqrt(v / (1 - 0.999^step)) + 1e-8)
    }
    if (optimizer %in% c("rprop", "grprop")) {
      product <- g * previous
      delta <- if (product > 0) min(delta * 1.2, 50) else if (product < 0) max(delta * 0.5, 1e-6) else delta
      previous <- if (product < 0) 0 else g
      value <- value - sign(previous) * delta
    }
    result <- get(".optimizer_step")(params, list(W = list(matrix(g, 1, 1)), b = list(g)), state,
                                     optimizer, 0.02, 0.9, 0.999, 1e-8, step, 0.8)
    params <- result$params
    state <- result$state
    near(params$W[[1]], value, 1e-10)
    near(params$b[[1]], value, 1e-10)
  }
}
cat("PASS:", count, "finite-difference gradient cases; saturated losses, clipping and 24 optimizer steps.\n")
