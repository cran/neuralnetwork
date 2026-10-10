.init_params <- function(layers, activation) {
  n_layers <- length(layers) - 1L
  weights <- vector("list", n_layers)
  biases <- vector("list", n_layers)

  for (i in seq_len(n_layers)) {
    fan_in <- layers[i]
    fan_out <- layers[i + 1L]
    scale <- if (activation %in% c("relu", "leaky_relu")) {
      sqrt(2 / max(1L, fan_in))
    } else {
      sqrt(1 / max(1L, fan_in))
    }
    weights[[i]] <- matrix(stats::rnorm(fan_in * fan_out, sd = scale),
                           nrow = fan_in, ncol = fan_out)
    biases[[i]] <- rep(0, fan_out)
  }

  list(W = weights, b = biases)
}

.copy_params <- function(params) {
  list(
    W = lapply(params$W, function(x) x + 0),
    b = lapply(params$b, function(x) x + 0)
  )
}

.copy_optimizer_state <- function(state) {
  lapply(state, function(x) {
    if (is.list(x)) {
      lapply(x, function(y) y + 0)
    } else {
      x
    }
  })
}

.shrink_rprop_state <- function(state, factor = 0.5, step_min = 1e-6) {
  state <- .copy_optimizer_state(state)
  state$stepW <- lapply(state$stepW, function(x) pmax(x * factor, step_min))
  state$stepb <- lapply(state$stepb, function(x) pmax(x * factor, step_min))
  state$prevW <- lapply(state$prevW, function(x) x * 0)
  state$prevb <- lapply(state$prevb, function(x) x * 0)
  state
}

.init_optimizer_state <- function(params, optimizer = NULL) {
  fields <- if (is.null(optimizer)) {
    c("mW", "vW", "mb", "vb", "velocityW", "velocityb",
      "prevW", "prevb", "stepW", "stepb")
  } else {
    switch(optimizer,
      adam = c("mW", "vW", "mb", "vb"),
      momentum = c("velocityW", "velocityb"),
      nesterov = c("velocityW", "velocityb"),
      rprop = c("prevW", "prevb", "stepW", "stepb"),
      grprop = c("prevW", "prevb", "stepW", "stepb"),
      sgd = character(), lbfgs = character(),
      stop("Unknown optimizer.", call. = FALSE)
    )
  }
  state <- list()
  for (field in fields) {
    template <- if (endsWith(field, "W")) params$W else params$b
    initial <- if (startsWith(field, "step")) 0.1 else 0
    state[[field]] <- lapply(template, function(x) x * 0 + initial)
  }
  state
}

.forward <- function(x, params, activation, task, dropout, training,
                     backend = "r") {
  n_layers <- length(params$W)
  caches <- vector("list", n_layers)
  a <- x

  for (i in seq_len(n_layers)) {
    z <- .dense(a, params$W[[i]], params$b[[i]], backend)
    a_prev <- a

    if (i < n_layers) {
      a <- .activate(z, activation, backend)
      mask <- NULL
      if (isTRUE(training) && dropout[i] > 0) {
        keep <- 1 - dropout[i]
        mask <- matrix(stats::runif(length(a)) < keep, nrow = nrow(a)) / keep
        a <- a * mask
      }
      caches[[i]] <- list(A_prev = a_prev, Z = z, mask = mask)
    } else {
      a <- if (task == "classification") {
        .softmax(z, backend)
      } else if (task == "binary_classification") {
        .sigmoid(z, backend)
      } else {
        z
      }
      caches[[i]] <- list(A_prev = a_prev, Z = z, mask = NULL)
    }
  }

  list(output = a, caches = caches)
}

.forward_trace <- function(x, params, activation, task, backend = "r") {
  n_layers <- length(params$W)
  a <- x
  neurons <- list(input = x)

  for (i in seq_len(n_layers)) {
    z <- .dense(a, params$W[[i]], params$b[[i]], backend)
    if (i < n_layers) {
      a <- .activate(z, activation, backend)
      neurons[[paste0("hidden", i)]] <- a
    } else {
      a <- if (task == "classification") {
        .softmax(z, backend)
      } else if (task == "binary_classification") {
        .sigmoid(z, backend)
      } else {
        z
      }
      neurons$output <- a
    }
  }

  list(neurons = neurons, output = a)
}

.backward <- function(forward, params, y, activation, task, l2,
                      weights = NULL, loss = "squared_error",
                      huber_delta = 1, regularization_denom = NULL,
                      training_n = NULL) {
  n_layers <- length(params$W)
  denom <- .weight_denom(weights, nrow(y))
  regularization_denom <- regularization_denom %||% denom
  row_scale <- .row_weights(weights, nrow(y)) / if (is.null(training_n)) {
    denom
  } else {
    regularization_denom
  }
  if (!is.null(training_n)) row_scale <- row_scale * (training_n / nrow(y))
  residual <- forward$output - y
  if (task == "regression" && loss == "huber") {
    residual <- pmax(pmin(residual, huber_delta), -huber_delta)
  }
  dZ <- residual * row_scale
  penalty_scale <- if (l2 > 0) sqrt(l2) / sqrt(regularization_denom) else 0
  grads_W <- vector("list", n_layers)
  grads_b <- vector("list", n_layers)

  for (i in n_layers:1L) {
    cache <- forward$caches[[i]]
    grads_W[[i]] <- crossprod(cache$A_prev, dZ)
    if (l2 > 0) {
      grads_W[[i]] <- grads_W[[i]] + (params$W[[i]] * penalty_scale) * penalty_scale
    }
    grads_b[[i]] <- colSums(dZ)

    if (i > 1L) {
      dA <- dZ %*% t(params$W[[i]])
      prev_cache <- forward$caches[[i - 1L]]
      if (!is.null(prev_cache$mask)) {
        dA <- dA * prev_cache$mask
      }
      dZ <- dA * .activation_derivative(prev_cache$Z, activation)
    }
  }

  list(W = grads_W, b = grads_b)
}

.optimizer_step <- function(params, grads, state, optimizer, learning_rate,
                            beta1, beta2, epsilon, step, momentum) {
  for (i in seq_along(params$W)) {
    if (optimizer == "sgd") {
      params$W[[i]] <- params$W[[i]] - learning_rate * grads$W[[i]]
      params$b[[i]] <- params$b[[i]] - learning_rate * grads$b[[i]]
    } else if (optimizer %in% c("momentum", "nesterov")) {
      prev_velocity_w <- state$velocityW[[i]]
      prev_velocity_b <- state$velocityb[[i]]
      velocity_w <- momentum * prev_velocity_w - learning_rate * grads$W[[i]]
      velocity_b <- momentum * prev_velocity_b - learning_rate * grads$b[[i]]
      state$velocityW[[i]] <- velocity_w
      state$velocityb[[i]] <- velocity_b
      if (optimizer == "nesterov") {
        params$W[[i]] <- params$W[[i]] -
          momentum * prev_velocity_w + (1 + momentum) * velocity_w
        params$b[[i]] <- params$b[[i]] -
          momentum * prev_velocity_b + (1 + momentum) * velocity_b
      } else {
        params$W[[i]] <- params$W[[i]] + velocity_w
        params$b[[i]] <- params$b[[i]] + velocity_b
      }
    } else if (optimizer %in% c("rprop", "grprop")) {
      w_step <- .rprop_step(params$W[[i]], grads$W[[i]], state$prevW[[i]],
                            state$stepW[[i]])
      b_step <- .rprop_step(params$b[[i]], grads$b[[i]], state$prevb[[i]],
                            state$stepb[[i]])
      params$W[[i]] <- w_step$param
      params$b[[i]] <- b_step$param
      state$prevW[[i]] <- w_step$prev_grad
      state$prevb[[i]] <- b_step$prev_grad
      state$stepW[[i]] <- w_step$step
      state$stepb[[i]] <- b_step$step
    } else {
      state$mW[[i]] <- beta1 * state$mW[[i]] + (1 - beta1) * grads$W[[i]]
      state$vW[[i]] <- beta2 * state$vW[[i]] + (1 - beta2) * grads$W[[i]]^2
      state$mb[[i]] <- beta1 * state$mb[[i]] + (1 - beta1) * grads$b[[i]]
      state$vb[[i]] <- beta2 * state$vb[[i]] + (1 - beta2) * grads$b[[i]]^2
      if (any(!is.finite(state$vW[[i]])) || any(!is.finite(state$vb[[i]]))) {
        stop("Optimizer produced non-finite second moments. Rescale the data or reduce the learning rate.", call. = FALSE)
      }

      mW_hat <- state$mW[[i]] / (1 - beta1^step)
      vW_hat <- state$vW[[i]] / (1 - beta2^step)
      mb_hat <- state$mb[[i]] / (1 - beta1^step)
      vb_hat <- state$vb[[i]] / (1 - beta2^step)

      params$W[[i]] <- params$W[[i]] -
        learning_rate * mW_hat / (sqrt(vW_hat) + epsilon)
      params$b[[i]] <- params$b[[i]] -
        learning_rate * mb_hat / (sqrt(vb_hat) + epsilon)
    }
  }

  if (!.all_finite(params)) {
    stop("Optimizer produced non-finite parameters or state. Rescale the data or reduce the learning rate.", call. = FALSE)
  }
  list(params = params, state = state)
}

.all_finite <- function(parts) {
  all(vapply(parts, function(x) {
    if (is.list(x)) .all_finite(x) else all(is.finite(x))
  }, logical(1)))
}

.rprop_step <- function(param, grad, prev_grad, step,
                        increase = 1.2, decrease = 0.5,
                        step_min = 1e-6, step_max = 50) {
  prod <- sign(grad) * sign(prev_grad)
  step <- ifelse(prod > 0, pmin(step * increase, step_max),
                 ifelse(prod < 0, pmax(step * decrease, step_min), step))
  adjusted_grad <- ifelse(prod < 0, 0, grad)
  param <- param - sign(adjusted_grad) * step
  list(param = param, prev_grad = adjusted_grad, step = step)
}

.dataset_loss <- function(x, y, params, activation, task, l2, backend = "r",
                          weights = NULL, loss = "squared_error",
                          huber_delta = 1) {
  forward <- .forward(x, params, activation, task, dropout = 0,
                      training = FALSE, backend = backend)
  data_loss <- .loss_from_output(forward$output, y, task, weights, loss,
                                huber_delta, forward$caches[[length(params$W)]]$Z)
  penalty <- .l2_penalty(params, l2, weights, nrow(y))
  data_loss + penalty
}

.dataset_metric <- function(x, y_raw, params, activation, task, backend = "r",
                            target_scaler = NULL, weights = NULL) {
  out <- .forward(x, params, activation, task, dropout = 0,
                  training = FALSE, backend = backend)$output
  .metric_from_output(out, y_raw, task, target_scaler, weights)
}

.dataset_stats <- function(x, y, y_raw, params, activation, task, l2,
                           backend = "r", target_scaler = NULL,
                           weights = NULL, loss = "squared_error",
                           huber_delta = 1) {
  forward <- .forward(x, params, activation, task, dropout = 0,
                      training = FALSE, backend = backend)
  out <- forward$output
  data_loss <- .loss_from_output(out, y, task, weights, loss, huber_delta,
                                forward$caches[[length(params$W)]]$Z)
  penalty <- .l2_penalty(params, l2, weights, nrow(y))
  metric <- .metric_from_output(out, y_raw, task, target_scaler, weights)
  list(loss = data_loss + penalty, metric = metric)
}

.l2_penalty <- function(params, l2, weights, n) {
  if (l2 == 0) return(0)
  multiplier <- sqrt(l2) / sqrt(.weight_denom(weights, n))
  sum(vapply(params$W, function(w) sum((w * multiplier)^2), numeric(1))) / 2
}

.loss_from_output <- function(out, y, task, weights = NULL,
                              loss = "squared_error", huber_delta = 1,
                              logits = NULL) {
  if (task == "classification") {
    row_loss <- if (is.null(logits)) {
      -rowSums(y * log(pmax(out, 1e-15)))
    } else {
      maximum <- apply(logits, 1L, max)
      shifted <- logits - maximum
      terms <- y * shifted
      extreme <- !is.finite(shifted) & y > 0
      if (any(extreme)) terms[extreme] <- (y * logits - y * maximum)[extreme]
      terms[y == 0] <- 0
      log(rowSums(exp(shifted))) - rowSums(terms)
    }
    return(.weighted_mean(row_loss, weights))
  }

  if (task == "binary_classification") {
    if (is.null(logits)) {
      out <- pmin(pmax(out, 1e-15), 1 - 1e-15)
      row_loss <- -(y[, 1L] * log(out[, 1L]) +
        (1 - y[, 1L]) * log(1 - out[, 1L]))
    } else {
      z <- logits[, 1L]
      linear <- numeric(length(z))
      positive <- z >= 0
      linear[positive] <- (1 - y[positive, 1L]) * z[positive]
      linear[!positive] <- -y[!positive, 1L] * z[!positive]
      row_loss <- linear + log1p(exp(-abs(z)))
    }
    return(.weighted_mean(row_loss, weights))
  }

  residual <- out - y
  if (loss == "huber") {
    abs_residual <- abs(residual)
    row_loss <- rowSums(ifelse(
      abs_residual <= huber_delta,
      0.5 * residual^2,
      huber_delta * (abs_residual - 0.5 * huber_delta)
    ))
  } else {
    row_loss <- 0.5 * rowSums(residual^2)
  }
  .weighted_mean(row_loss, weights)
}

.metric_from_output <- function(out, y_raw, task, target_scaler = NULL,
                                weights = NULL) {
  weights <- .row_weights(weights, nrow(y_raw))

  if (task == "classification") {
    truth <- max.col(y_raw, ties.method = "first")
    pred <- max.col(out, ties.method = "first")
    return(.weighted_mean(pred == truth, weights))
  }

  if (task == "binary_classification") {
    truth <- y_raw[, 1L] >= 0.5
    pred <- out[, 1L] >= 0.5
    return(.weighted_mean(pred == truth, weights))
  }

  out <- .inverse_target_scaler(out, target_scaler)
  .weighted_rmse(out - y_raw, weights)
}

.metric_name <- function(task) {
  if (task %in% c("classification", "binary_classification")) {
    "accuracy"
  } else {
    "rmse"
  }
}

.row_weights <- function(weights, n) {
  if (is.null(weights)) {
    return(rep(1, n))
  }
  weights
}

.weighted_mean <- function(x, weights) {
  w <- .row_weights(weights, length(x))
  w <- w / .weight_denom(w, length(x))
  positive <- w > 0
  sum(x[positive] * w[positive])
}

.weighted_rmse <- function(error, weights) {
  positive <- .row_weights(weights, nrow(error)) > 0
  scale <- max(abs(error[positive, , drop = FALSE]))
  if (!is.finite(scale)) stop("Regression residuals exceeded numerical range. Rescale values.", call. = FALSE)
  if (scale == 0) return(0)
  scale * sqrt(.weighted_mean(rowMeans((error / scale)^2), weights))
}

.weight_denom <- function(weights, n) {
  denom <- sum(.row_weights(weights, n))
  if (!is.finite(denom) || denom <= 0) {
    stop("Sample weights must have a positive finite sum.", call. = FALSE)
  }
  denom
}

.gradient_norm <- function(grads) {
  values <- .flatten_grads(grads)
  if (any(!is.finite(values))) {
    stop("Non-finite gradients encountered. Rescale the data or reduce the learning rate.",
         call. = FALSE)
  }
  largest <- max(abs(values))
  if (largest == 0) return(0)
  largest * sqrt(sum((values / largest)^2))
}

.clip_grads <- function(grads, gradient_clip, norm = .gradient_norm(grads)) {
  if (!is.finite(gradient_clip) || norm <= gradient_clip || norm == 0) {
    return(grads)
  }
  values <- .flatten_grads(grads)
  largest <- max(abs(values))
  scale <- (gradient_clip / largest) / sqrt(sum((values / largest)^2))
  grads$W <- lapply(grads$W, `*`, scale)
  grads$b <- lapply(grads$b, `*`, scale)
  grads
}

.dense <- function(x, w, b, backend = "r") {
  out <- if (backend == "rcpp") {
    .Call("neuralnetwork_dense_bias", x, w, b, PACKAGE = "neuralnetwork")
  } else {
    sweep(x %*% w, 2L, b, "+")
  }
  if (any(!is.finite(out))) {
    stop("Dense layer produced non-finite values. Rescale the data or reduce the learning rate.",
         call. = FALSE)
  }
  out
}

.activate <- function(x, activation, backend = "r") {
  if (backend == "rcpp") {
    return(.Call("neuralnetwork_activate", x, activation, PACKAGE = "neuralnetwork"))
  }
  switch(
    activation,
    relu = pmax(x, 0),
    leaky_relu = ifelse(x > 0, x, 0.01 * x),
    tanh = tanh(x),
    sigmoid = .stable_sigmoid(x)
  )
}

.activation_derivative <- function(x, activation) {
  switch(
    activation,
    relu = ifelse(x > 0, 1, 0),
    leaky_relu = ifelse(x > 0, 1, 0.01),
    tanh = 1 - tanh(x)^2,
    sigmoid = {
      s <- .stable_sigmoid(x)
      s * (1 - s)
    }
  )
}

.stable_sigmoid <- function(x) {
  out <- x
  positive <- !is.na(x) & x >= 0
  negative <- !is.na(x) & x < 0
  out[positive] <- 1 / (1 + exp(-x[positive]))
  ex <- exp(x[negative])
  out[negative] <- ex / (1 + ex)
  out
}

.sigmoid <- function(x, backend = "r") {
  if (backend == "rcpp") {
    return(.Call("neuralnetwork_sigmoid", x, PACKAGE = "neuralnetwork"))
  }
  .stable_sigmoid(x)
}

.softmax <- function(x, backend = "r") {
  if (ncol(x) == 0L || any(!is.finite(x))) {
    stop("Softmax requires finite values and at least one output column.",
         call. = FALSE)
  }
  if (nrow(x) == 0L) return(x)
  if (backend == "rcpp") {
    return(.Call("neuralnetwork_softmax", x, PACKAGE = "neuralnetwork"))
  }
  shifted <- x - apply(x, 1L, max)
  ex <- exp(shifted)
  ex / rowSums(ex)
}
