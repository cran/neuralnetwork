.init_params <- function(layers, activation) {
  n_layers <- length(layers) - 1L
  weights <- vector("list", n_layers)
  biases <- vector("list", n_layers)

  for (i in seq_len(n_layers)) {
    fan_in <- layers[i]
    fan_out <- layers[i + 1L]
    scale <- if (activation %in% c("relu", "leaky_relu")) {
      sqrt(2 / fan_in)
    } else {
      sqrt(1 / fan_in)
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

.init_optimizer_state <- function(params) {
  list(
    mW = lapply(params$W, function(x) x * 0),
    vW = lapply(params$W, function(x) x * 0),
    mb = lapply(params$b, function(x) x * 0),
    vb = lapply(params$b, function(x) x * 0),
    velocityW = lapply(params$W, function(x) x * 0),
    velocityb = lapply(params$b, function(x) x * 0),
    prevW = lapply(params$W, function(x) x * 0),
    prevb = lapply(params$b, function(x) x * 0),
    stepW = lapply(params$W, function(x) x * 0 + 0.1),
    stepb = lapply(params$b, function(x) x * 0 + 0.1)
  )
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
                      huber_delta = 1) {
  n_layers <- length(params$W)
  denom <- .weight_denom(weights, nrow(y))
  row_scale <- .row_weights(weights, nrow(y)) / denom
  residual <- forward$output - y
  if (task == "regression" && loss == "huber") {
    residual <- pmax(pmin(residual, huber_delta), -huber_delta)
  }
  dZ <- residual * row_scale
  grads_W <- vector("list", n_layers)
  grads_b <- vector("list", n_layers)

  for (i in n_layers:1L) {
    cache <- forward$caches[[i]]
    grads_W[[i]] <- crossprod(cache$A_prev, dZ) + (l2 / denom) * params$W[[i]]
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

  list(params = params, state = state)
}

.rprop_step <- function(param, grad, prev_grad, step,
                        increase = 1.2, decrease = 0.5,
                        step_min = 1e-6, step_max = 50) {
  prod <- grad * prev_grad
  step <- ifelse(prod > 0, pmin(step * increase, step_max),
                 ifelse(prod < 0, pmax(step * decrease, step_min), step))
  adjusted_grad <- ifelse(prod < 0, 0, grad)
  param <- param - sign(adjusted_grad) * step
  list(param = param, prev_grad = adjusted_grad, step = step)
}

.dataset_loss <- function(x, y, params, activation, task, l2, backend = "r",
                          weights = NULL, loss = "squared_error",
                          huber_delta = 1) {
  out <- .forward(x, params, activation, task, dropout = 0,
                  training = FALSE, backend = backend)$output
  data_loss <- .loss_from_output(out, y, task, weights, loss, huber_delta)
  penalty <- 0.5 * l2 * sum(vapply(params$W, function(w) sum(w^2), numeric(1))) /
    .weight_denom(weights, nrow(y))
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
  out <- .forward(x, params, activation, task, dropout = 0,
                  training = FALSE, backend = backend)$output
  data_loss <- .loss_from_output(out, y, task, weights, loss, huber_delta)
  penalty <- 0.5 * l2 * sum(vapply(params$W, function(w) sum(w^2), numeric(1))) /
    .weight_denom(weights, nrow(y))
  metric <- .metric_from_output(out, y_raw, task, target_scaler, weights)
  list(loss = data_loss + penalty, metric = metric)
}

.loss_from_output <- function(out, y, task, weights = NULL,
                              loss = "squared_error", huber_delta = 1) {
  if (task == "classification") {
    row_loss <- -rowSums(y * log(pmax(out, 1e-15)))
    return(stats::weighted.mean(row_loss, .row_weights(weights, nrow(y))))
  }

  if (task == "binary_classification") {
    out <- pmin(pmax(out, 1e-15), 1 - 1e-15)
    row_loss <- -(y[, 1L] * log(out[, 1L]) +
      (1 - y[, 1L]) * log(1 - out[, 1L]))
    return(stats::weighted.mean(row_loss, .row_weights(weights, nrow(y))))
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
  stats::weighted.mean(row_loss, .row_weights(weights, nrow(y)))
}

.metric_from_output <- function(out, y_raw, task, target_scaler = NULL,
                                weights = NULL) {
  weights <- .row_weights(weights, nrow(y_raw))

  if (task == "classification") {
    truth <- max.col(y_raw, ties.method = "first")
    pred <- max.col(out, ties.method = "first")
    return(stats::weighted.mean(pred == truth, weights))
  }

  if (task == "binary_classification") {
    truth <- y_raw[, 1L] >= 0.5
    pred <- out[, 1L] >= 0.5
    return(stats::weighted.mean(pred == truth, weights))
  }

  out <- .inverse_target_scaler(out, target_scaler)
  err <- rowSums((out - y_raw)^2)
  sqrt(stats::weighted.mean(err, weights))
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

.weight_denom <- function(weights, n) {
  denom <- sum(.row_weights(weights, n))
  if (!is.finite(denom) || denom <= 0) {
    stop("Sample weights must have a positive finite sum.", call. = FALSE)
  }
  denom
}

.gradient_norm <- function(grads) {
  sqrt(
    sum(vapply(grads$W, function(x) sum(x^2), numeric(1))) +
      sum(vapply(grads$b, function(x) sum(x^2), numeric(1)))
  )
}

.clip_grads <- function(grads, gradient_clip, norm = .gradient_norm(grads)) {
  if (!is.finite(gradient_clip) || norm <= gradient_clip || norm == 0) {
    return(grads)
  }
  scale <- gradient_clip / norm
  grads$W <- lapply(grads$W, `*`, scale)
  grads$b <- lapply(grads$b, `*`, scale)
  grads
}

.dense <- function(x, w, b, backend = "r") {
  if (backend == "rcpp") {
    return(.Call("neuralnetwork_dense_bias", x, w, b, PACKAGE = "neuralnetwork"))
  }
  sweep(x %*% w, 2L, b, "+")
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
  if (backend == "rcpp") {
    return(.Call("neuralnetwork_softmax", x, PACKAGE = "neuralnetwork"))
  }
  shifted <- x - apply(x, 1L, max)
  ex <- exp(shifted)
  ex / rowSums(ex)
}
