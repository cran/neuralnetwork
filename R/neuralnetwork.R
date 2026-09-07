nn_fit <- function(x,
                   y = NULL,
                   data = NULL,
                   hidden = "auto",
                   task = c("auto", "classification", "regression"),
                   activation = c("auto", "relu", "tanh", "sigmoid", "leaky_relu"),
                   optimizer = c("auto", "adam", "sgd", "momentum", "nesterov",
                                 "rprop", "grprop", "lbfgs"),
                   backend = c("auto", "rcpp", "r"),
                   loss = c("auto", "squared_error", "huber"),
                   huber_delta = 1,
                   epochs = 100,
                   stepmax = NULL,
                   batch_size = 32,
                   learning_rate = 0.001,
                   momentum = 0.9,
                   beta1 = 0.9,
                   beta2 = 0.999,
                   epsilon = 1e-8,
                   l2 = 0,
                   dropout = 0,
                   sample_weight = NULL,
                   class_weight = NULL,
                   gradient_clip = Inf,
                   learning_rate_decay = 0,
                   threshold = 0,
                   validation_split = 0,
                   patience = Inf,
                   min_delta = 1e-4,
                   scale = TRUE,
                   y_scale = TRUE,
                   shuffle = TRUE,
                   seed = NULL,
                   callbacks = NULL,
                   verbose = TRUE) {
  task <- match.arg(task)
  activation <- match.arg(activation)
  optimizer <- match.arg(optimizer)
  backend <- .resolve_backend(match.arg(backend))
  loss <- match.arg(loss)
  callbacks <- .validate_callbacks(callbacks)
  .validate_flag(scale, "scale")
  .validate_flag(y_scale, "y_scale")
  .validate_flag(shuffle, "shuffle")
  .validate_flag(verbose, "verbose")
  .validate_seed(seed)
  hidden <- .validate_hidden(hidden)
  epochs <- .resolve_epochs(epochs, stepmax)
  .validate_training_args(epochs, batch_size, learning_rate, l2,
                          validation_split, gradient_clip,
                          learning_rate_decay, threshold, momentum,
                          beta1, beta2, epsilon,
                          patience, min_delta)

  if (!is.null(seed)) {
    set.seed(seed)
  }

  prepared <- .prepare_xy(x, y, data)
  encoded <- .encode_y(prepared$y, task)
  task <- encoded$task
  loss <- .resolve_loss(loss, task, huber_delta)
  output_dim <- ncol(encoded$y)
  hidden <- .resolve_hidden(hidden, task, prepared$x, output_dim)
  dropout <- .validate_dropout(dropout, length(hidden))
  weights <- .make_sample_weights(encoded, sample_weight, class_weight)
  activation <- .resolve_activation(activation, task, prepared$x, hidden)
  optimizer <- .resolve_optimizer(optimizer, task, prepared$x, hidden, dropout,
                                  output_dim, callbacks)

  split <- .make_validation_split(nrow(prepared$x), validation_split)
  x_train_raw <- prepared$x[split$train, , drop = FALSE]
  y_train_raw <- encoded$y[split$train, , drop = FALSE]
  weight_train <- weights[split$train]
  x_val_raw <- prepared$x[split$validation, , drop = FALSE]
  y_val_raw <- encoded$y[split$validation, , drop = FALSE]
  weight_val <- weights[split$validation]

  target_scaler <- .fit_target_scaler(y_train_raw, task, y_scale)
  y_train <- .apply_target_scaler(y_train_raw, target_scaler)
  y_val <- .apply_target_scaler(y_val_raw, target_scaler)

  scaler <- .fit_scaler(x_train_raw, scale)
  x_train <- .apply_scaler(x_train_raw, scaler)
  x_val <- .apply_scaler(x_val_raw, scaler)

  layers <- c(ncol(x_train), hidden, output_dim)
  params <- .init_params(layers, activation)
  state <- .init_optimizer_state(params)

  history <- .new_history(0L)

  best_params <- .copy_params(params)
  best_loss <- Inf
  best_epoch <- 0L
  stale_epochs <- 0L
  step <- 0L
  stopped_early <- FALSE
  stop_reason <- NA_character_
  previous_monitor_loss <- Inf
  convergence <- list(code = NA_integer_, message = NA_character_)

  if (optimizer == "lbfgs") {
    if (any(dropout > 0)) {
      stop("`optimizer = \"lbfgs\"` requires `dropout = 0`.", call. = FALSE)
    }
    fit <- .fit_lbfgs(params, x_train, y_train, y_train_raw,
                      x_val, y_val, y_val_raw, activation, task, l2,
                      loss, huber_delta, backend, target_scaler, weight_train,
                      weight_val, maxit = epochs, verbose = verbose)
    params <- fit$params
    history <- fit$history
    best_params <- .copy_params(params)
    best_loss <- fit$best_loss
    best_epoch <- 1L
    convergence <- list(
      code = as.integer(fit$convergence),
      message = fit$message %||% NA_character_
    )
  } else {
    history <- .new_history(epochs)
    history_rows <- 0L
    for (epoch in seq_len(epochs)) {
      epoch_start_params <- if (optimizer == "grprop") {
        .copy_params(params)
      } else {
        NULL
      }
      epoch_start_state <- if (optimizer == "grprop") {
        .copy_optimizer_state(state)
      } else {
        NULL
      }
      order <- if (shuffle) sample.int(nrow(x_train)) else seq_len(nrow(x_train))
      batch_starts <- if (optimizer %in% c("rprop", "grprop")) {
        1L
      } else {
        seq.int(1L, length(order), by = batch_size)
      }
      epoch_grad_norm <- 0
      backtracked <- FALSE

      for (batch_start in batch_starts) {
        batch <- if (optimizer %in% c("rprop", "grprop")) {
          order
        } else {
          batch_end <- min(batch_start + batch_size - 1L, length(order))
          order[batch_start:batch_end]
        }
        step <- step + 1L
        xb <- x_train[batch, , drop = FALSE]
        yb <- y_train[batch, , drop = FALSE]
        wb <- weight_train[batch]

        forward <- .forward(xb, params, activation, task, dropout,
                            training = TRUE, backend = backend)
        grads <- .backward(forward, params, yb, activation, task, l2,
                           weights = wb, loss = loss,
                           huber_delta = huber_delta)
        grad_norm <- .gradient_norm(grads)
        epoch_grad_norm <- max(epoch_grad_norm, grad_norm)
        grads <- .clip_grads(grads, gradient_clip, grad_norm)
        updated <- .optimizer_step(params, grads, state, optimizer, learning_rate,
                                   beta1, beta2, epsilon, step, momentum)
        params <- updated$params
        state <- updated$state
      }

      train_stats <- .dataset_stats(x_train, y_train, y_train_raw, params,
                                    activation, task, l2, backend,
                                    target_scaler, weight_train, loss,
                                    huber_delta)
      train_loss <- train_stats$loss
      train_metric <- train_stats$metric

      validation_stats <- if (length(split$validation) > 0L) {
        .dataset_stats(x_val, y_val, y_val_raw, params, activation, task, l2,
                       backend, target_scaler, weight_val, loss, huber_delta)
      } else {
        list(loss = NA_real_, metric = NA_real_)
      }
      validation_loss <- validation_stats$loss
      validation_metric <- validation_stats$metric

      monitor_loss <- if (is.na(validation_loss)) train_loss else validation_loss

      if (optimizer == "grprop" && monitor_loss > previous_monitor_loss) {
        params <- epoch_start_params
        state <- .shrink_rprop_state(epoch_start_state)
        backtracked <- TRUE
        train_stats <- .dataset_stats(x_train, y_train, y_train_raw, params,
                                      activation, task, l2, backend,
                                      target_scaler, weight_train, loss,
                                      huber_delta)
        train_loss <- train_stats$loss
        train_metric <- train_stats$metric
        validation_stats <- if (length(split$validation) > 0L) {
          .dataset_stats(x_val, y_val, y_val_raw, params, activation, task, l2,
                         backend, target_scaler, weight_val, loss,
                         huber_delta)
        } else {
          list(loss = NA_real_, metric = NA_real_)
        }
        validation_loss <- validation_stats$loss
        validation_metric <- validation_stats$metric
        monitor_loss <- if (is.na(validation_loss)) train_loss else validation_loss
      }

      previous_monitor_loss <- monitor_loss

      history_rows <- history_rows + 1L
      history[history_rows, ] <- list(
        epoch = epoch,
        train_loss = train_loss,
        validation_loss = validation_loss,
        train_metric = train_metric,
        validation_metric = validation_metric,
        gradient_norm = epoch_grad_norm,
        learning_rate = learning_rate,
        backtracked = backtracked
      )

      if (monitor_loss < best_loss - min_delta) {
        best_loss <- monitor_loss
        best_epoch <- epoch
        best_params <- .copy_params(params)
        stale_epochs <- 0L
      } else {
        stale_epochs <- stale_epochs + 1L
      }

      if (isTRUE(verbose) && .should_report(epoch, epochs)) {
        .report_epoch(epoch, train_loss, validation_loss)
      }

      callback_state <- list(
        epoch = epoch,
        train_loss = train_loss,
        validation_loss = validation_loss,
        train_metric = train_metric,
        validation_metric = validation_metric,
        best_loss = best_loss,
        best_epoch = best_epoch,
        gradient_norm = epoch_grad_norm,
        learning_rate = learning_rate,
        task = task
      )
      callback_result <- .run_callbacks(callbacks, callback_state)
      learning_rate <- callback_result$learning_rate
      if (isTRUE(callback_result$stop)) {
        stopped_early <- TRUE
        stop_reason <- "callback"
        break
      }

      if (threshold > 0 && epoch_grad_norm <= threshold) {
        stopped_early <- TRUE
        stop_reason <- "threshold"
        break
      }

      if (is.finite(patience) && stale_epochs >= patience) {
        stopped_early <- TRUE
        stop_reason <- "patience"
        break
      }

      if (learning_rate_decay > 0) {
        learning_rate <- learning_rate / (1 + learning_rate_decay)
      }
    }

    history <- history[seq_len(history_rows), , drop = FALSE]
    params <- best_params
  }

  model <- list(
    params = params,
    task = task,
    activation = activation,
    hidden = hidden,
    optimizer = optimizer,
    loss = loss,
    huber_delta = huber_delta,
    momentum = if (optimizer %in% c("momentum", "nesterov")) momentum else NULL,
    threshold = threshold,
    backend = backend,
    scaler = scaler,
    target_scaler = target_scaler,
    validation = if (nrow(x_val) > 0L) {
      list(x = x_val, y = y_val, y_raw = y_val_raw)
    } else {
      NULL
    },
    blueprint = prepared$blueprint,
    classes = encoded$classes,
    output_names = encoded$output_names,
    output_dim = output_dim,
    history = history,
    epochs = if (optimizer == "lbfgs") history$epoch[nrow(history)] else nrow(history),
    training_units = if (optimizer == "lbfgs") "function_evaluations" else "epochs",
    best_epoch = best_epoch,
    stopped_early = stopped_early,
    stop_reason = stop_reason,
    convergence = convergence,
    metric_name = .metric_name(task),
    call = match.call()
  )
  class(model) <- "neuralnetwork"
  model
}

predict.neuralnetwork <- function(object,
                             newdata,
                             type = c("response", "class", "prob"),
                             ...) {
  type <- match.arg(type)
  if (missing(newdata)) {
    stop("`newdata` is required.", call. = FALSE)
  }

  x <- .prepare_new_x(newdata, object$blueprint)
  x <- .apply_scaler(x, object$scaler)
  out <- .forward(x, object$params, object$activation, object$task,
                  dropout = 0, training = FALSE,
                  backend = object$backend %||% "r")$output

  if (object$task == "binary_classification") {
    p <- as.numeric(out[, 1L])
    prob <- cbind(1 - p, p)
    colnames(prob) <- object$classes
    if (type == "prob") {
      return(prob)
    }
    if (type %in% c("class", "response")) {
      idx <- ifelse(p >= 0.5, 2L, 1L)
      return(factor(object$classes[idx], levels = object$classes))
    }
  }

  if (object$task == "classification") {
    colnames(out) <- object$classes
    if (type == "prob") {
      return(out)
    }
    if (type %in% c("class", "response")) {
      idx <- max.col(out, ties.method = "first")
      return(factor(object$classes[idx], levels = object$classes))
    }
  }

  if (type %in% c("class", "prob")) {
    stop("`type = \"class\"` and `type = \"prob\"` are only valid for classification.",
         call. = FALSE)
  }

  colnames(out) <- object$output_names
  out <- .inverse_target_scaler(out, object$target_scaler %||% NULL)
  if (ncol(out) == 1L) {
    as.numeric(out[, 1L])
  } else {
    out
  }
}

print.neuralnetwork <- function(x, ...) {
  layer_sizes <- c(length(x$blueprint$feature_names), x$hidden,
                   x$output_dim %||% length(x$output_names))
  final <- .final_history(x)
  cat("neuralnetwork model\n")
  cat("  Task:       ", .task_label(x$task), "\n", sep = "")
  cat("  Layers:     ", paste(layer_sizes, collapse = " -> "), "\n", sep = "")
  cat("  Optimizer:  ", x$optimizer, " | activation: ", x$activation,
      " | backend: ", x$backend %||% "r", "\n", sep = "")
  cat("  Loss:       ", .loss_label(x), "\n", sep = "")
  if (identical(x$training_units %||% "epochs", "function_evaluations")) {
    cat("  Trained:    ", x$epochs, " function evaluation",
        if (x$epochs == 1L) "" else "s", "\n", sep = "")
    if (!is.null(x$convergence) && !is.na(x$convergence$code)) {
      message <- x$convergence$message
      message <- if (length(message) && !is.na(message)) {
        paste0(" | message: ", message)
      } else {
        ""
      }
      cat("  Convergence: code ", x$convergence$code, message, "\n", sep = "")
    }
  } else {
    cat("  Trained:    ", x$epochs, " epoch", if (x$epochs == 1L) "" else "s",
        " | best epoch: ", x$best_epoch, "\n", sep = "")
  }
  cat("  Final:      train loss ", .fmt(final$train_loss),
      " | ", x$metric_name, " ", .fmt(final$train_metric), "\n", sep = "")
  if (!is.na(final$validation_loss)) {
    cat("              validation loss ", .fmt(final$validation_loss),
        " | validation ", x$metric_name, " ",
        .fmt(final$validation_metric), "\n", sep = "")
  }
  if (isTRUE(x$stopped_early)) {
    cat("  Stopped:    ", x$stop_reason %||% "early", "\n", sep = "")
  }
  invisible(x)
}

plot.neuralnetwork <- function(x,
                          y = NULL,
                          type = c("loss", "network"),
                          main = "Training loss",
                          xlab = "Epoch",
                          ylab = "Loss",
                          ...) {
  type <- match.arg(type)
  if (type == "network") {
    return(.plot_network(x, ...))
  }
  history <- x$history
  graphics::plot(history$epoch, history$train_loss, type = "l",
                 main = main, xlab = xlab, ylab = ylab, ...)
  if (any(!is.na(history$validation_loss))) {
    graphics::lines(history$epoch, history$validation_loss, col = 2)
    graphics::legend("topright", legend = c("train", "validation"),
                     col = c(1, 2), lty = 1, bty = "n")
  }
  invisible(x)
}

nn_save <- function(model, path, compress = TRUE) {
  if (!inherits(model, "neuralnetwork")) {
    stop("`model` must be a neuralnetwork object.", call. = FALSE)
  }
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    stop("`path` must be a non-empty file path.", call. = FALSE)
  }
  parent <- dirname(path)
  if (!dir.exists(parent)) {
    stop("The directory for `path` does not exist: ", parent, call. = FALSE)
  }
  saveRDS(model, file = path, compress = compress)
  invisible(normalizePath(path, mustWork = FALSE))
}

nn_load <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    stop("`path` must be a non-empty file path.", call. = FALSE)
  }
  if (!file.exists(path)) {
    stop("`path` does not exist: ", path, call. = FALSE)
  }
  model <- readRDS(path)
  if (!inherits(model, "neuralnetwork")) {
    stop("`path` does not contain a neuralnetwork model.", call. = FALSE)
  }
  model
}

nn_class_ind <- function(classes) {
  classes <- if (is.factor(classes)) classes else factor(classes)
  if (!length(classes)) {
    stop("`classes` must contain at least one value.", call. = FALSE)
  }
  if (anyNA(classes)) {
    stop("`classes` cannot contain missing values.", call. = FALSE)
  }
  out <- matrix(0, nrow = length(classes), ncol = nlevels(classes))
  out[cbind(seq_along(classes), as.integer(classes))] <- 1
  colnames(out) <- levels(classes)
  out
}

nn_which_is_max <- function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    return(max.col(as.matrix(x), ties.method = "first"))
  }
  which.max(x)
}

nn_multinom <- function(formula, data, weights = NULL, ...) {
  nn_fit(
    formula,
    data = data,
    hidden = integer(0),
    task = "classification",
    sample_weight = weights,
    ...
  )
}

nn_compute <- function(model, newdata) {
  if (!inherits(model, "neuralnetwork")) {
    stop("`model` must be a neuralnetwork object.", call. = FALSE)
  }
  if (missing(newdata)) {
    stop("`newdata` is required.", call. = FALSE)
  }
  x <- .prepare_new_x(newdata, model$blueprint)
  x <- .apply_scaler(x, model$scaler)
  traced <- .forward_trace(x, model$params, model$activation, model$task,
                           model$backend %||% "r")
  net_result <- traced$output
  if (model$task == "binary_classification") {
    p <- as.numeric(net_result[, 1L])
    net_result <- cbind(1 - p, p)
    colnames(net_result) <- model$classes
  } else if (model$task == "classification") {
    colnames(net_result) <- model$classes
  } else {
    net_result <- .inverse_target_scaler(net_result, model$target_scaler %||% NULL)
    colnames(net_result) <- model$output_names
  }
  list(neurons = traced$neurons, net.result = net_result)
}

nn_generalized_weights <- function(model, newdata, response = 1L,
                                   epsilon = 1e-4) {
  if (!inherits(model, "neuralnetwork")) {
    stop("`model` must be a neuralnetwork object.", call. = FALSE)
  }
  if (missing(newdata)) {
    stop("`newdata` is required.", call. = FALSE)
  }
  .validate_positive_scalar(epsilon, "epsilon")
  x <- .prepare_new_x(newdata, model$blueprint)
  x <- .apply_scaler(x, model$scaler)
  response <- .resolve_response_index(model, response)
  base <- .predict_matrix_response(model, x, response)
  out <- matrix(NA_real_, nrow = nrow(x), ncol = ncol(x))
  colnames(out) <- model$blueprint$feature_names
  for (j in seq_len(ncol(x))) {
    xp <- x
    xp[, j] <- xp[, j] + epsilon
    out[, j] <- (.predict_matrix_response(model, xp, response) - base) / epsilon
  }
  out
}

nn_gwplot <- function(model, newdata, selected.covariate = 1L,
                      response = 1L, ...) {
  if (missing(newdata)) {
    stop("`newdata` is required.", call. = FALSE)
  }
  gw <- nn_generalized_weights(model, newdata, response = response)
  feature <- .resolve_feature_index(model, selected.covariate)
  x <- .prepare_new_x(newdata, model$blueprint)[, feature]
  graphics::plot(x, gw[, feature], xlab = colnames(gw)[feature],
                 ylab = "Generalized weight", ...)
  invisible(gw)
}

nn_hessian <- function(model, newdata, y = NULL, epsilon = 1e-4,
                       max_params = 120) {
  if (!inherits(model, "neuralnetwork")) {
    stop("`model` must be a neuralnetwork object.", call. = FALSE)
  }
  if (missing(newdata)) {
    stop("`newdata` is required.", call. = FALSE)
  }
  .validate_positive_scalar(epsilon, "epsilon")
  .validate_positive_integer(max_params, "max_params")
  x <- .prepare_new_x(newdata, model$blueprint)
  x <- .apply_scaler(x, model$scaler)
  truth_raw <- .extract_truth(model, newdata, y)
  .check_truth_length(truth_raw, nrow(x))
  truth <- .encode_truth_for_model(model, truth_raw)
  flat <- .flatten_params(model$params)
  p <- length(flat$values)
  if (p > max_params) {
    stop("Model has ", p, " parameters; increase `max_params` to compute a finite-difference Hessian.",
         call. = FALSE)
  }

  loss_at <- function(values) {
    params <- .unflatten_params(values, model$params)
    .dataset_loss(x, truth, params, model$activation, model$task, l2 = 0,
                  backend = model$backend %||% "r",
                  loss = model$loss %||% "squared_error",
                  huber_delta = model$huber_delta %||% 1)
  }

  hessian <- matrix(0, nrow = p, ncol = p,
                    dimnames = list(flat$names, flat$names))
  for (i in seq_len(p)) {
    for (j in i:p) {
      ei <- rep(0, p)
      ej <- rep(0, p)
      ei[i] <- epsilon
      ej[j] <- epsilon
      value <- (loss_at(flat$values + ei + ej) -
        loss_at(flat$values + ei - ej) -
        loss_at(flat$values - ei + ej) +
        loss_at(flat$values - ei - ej)) / (4 * epsilon^2)
      hessian[i, j] <- value
      hessian[j, i] <- value
    }
  }
  hessian
}

nn_confint <- function(model, newdata, y = NULL, level = 0.95,
                       epsilon = 1e-4, max_params = 120) {
  if (missing(newdata)) {
    stop("`newdata` is required.", call. = FALSE)
  }
  .validate_probability_level(level)
  hessian <- nn_hessian(model, newdata, y = y, epsilon = epsilon,
                        max_params = max_params)
  flat <- .flatten_params(model$params)
  vcov <- tryCatch(solve(hessian), error = function(e) NULL)
  if (is.null(vcov)) {
    vcov <- tryCatch(solve(hessian + diag(1e-6, nrow(hessian))),
                     error = function(e) NULL)
  }
  if (is.null(vcov)) {
    stop("Could not invert Hessian; confidence intervals are not available for this fit.",
         call. = FALSE)
  }
  se <- sqrt(pmax(diag(vcov), 0))
  alpha <- 1 - level
  z <- stats::qnorm(1 - alpha / 2)
  data.frame(
    term = flat$names,
    estimate = flat$values,
    std.error = se,
    conf.low = flat$values - z * se,
    conf.high = flat$values + z * se,
    row.names = NULL
  )
}

coef.neuralnetwork <- function(object, ...) {
  object$params
}

summary.neuralnetwork <- function(object, ...) {
  structure(list(
    task = object$task,
    layers = c(length(object$blueprint$feature_names), object$hidden,
               object$output_dim),
    activation = object$activation,
    optimizer = object$optimizer,
    loss = object$loss %||% if (object$task == "regression") {
      "squared_error"
    } else {
      "cross_entropy"
    },
    huber_delta = object$huber_delta %||% NA_real_,
    backend = object$backend,
    epochs = object$epochs,
    training_units = object$training_units %||% "epochs",
    best_epoch = object$best_epoch,
    metric_name = object$metric_name,
    convergence = object$convergence %||% list(
      code = NA_integer_,
      message = NA_character_
    ),
    final = object$history[nrow(object$history), , drop = FALSE]
  ), class = "summary.neuralnetwork")
}

print.summary.neuralnetwork <- function(x, ...) {
  cat("neuralnetwork summary\n")
  cat("  Task:       ", .task_label(x$task), "\n", sep = "")
  cat("  Layers:     ", paste(x$layers, collapse = " -> "), "\n", sep = "")
  cat("  Optimizer:  ", x$optimizer, " | activation: ", x$activation,
      " | backend: ", x$backend, "\n", sep = "")
  cat("  Loss:       ", if (identical(x$loss, "huber")) {
    paste0("huber (delta=", .fmt(x$huber_delta), ")")
  } else {
    x$loss
  }, "\n", sep = "")
  if (identical(x$training_units %||% "epochs", "function_evaluations")) {
    cat("  Evaluations: ", x$epochs, "\n", sep = "")
  } else {
    cat("  Epochs:     ", x$epochs, " | best epoch: ", x$best_epoch, "\n", sep = "")
  }
  if (identical(x$optimizer, "lbfgs")) {
    message <- x$convergence$message
    message <- if (length(message) && !is.na(message)) {
      paste0(" | message: ", message)
    } else {
      ""
    }
    cat("  Convergence: code ", x$convergence$code, message, "\n", sep = "")
  }
  cat("\nFinal training row:\n")
  print(.format_history_row(x$final), row.names = FALSE)
  invisible(x)
}

nn_evaluate <- function(model, newdata, y = NULL) {
  if (!inherits(model, "neuralnetwork")) {
    stop("`model` must be a neuralnetwork object.", call. = FALSE)
  }
  if (missing(newdata)) {
    stop("`newdata` is required.", call. = FALSE)
  }

  truth <- .extract_truth(model, newdata, y)

  if (model$task %in% c("classification", "binary_classification")) {
    pred <- predict(model, newdata, type = "class")
    .check_truth_length(truth, NROW(pred))
    prob <- predict(model, newdata, type = "prob")
    truth <- factor(truth, levels = model$classes)
    confusion <- table(truth = truth, estimate = pred)
    metrics <- .classification_metrics(truth, pred, prob, model$classes)
    return(structure(list(
      metrics = metrics,
      confusion = confusion,
      predictions = pred,
      probabilities = prob
    ), class = "neuralnetwork_evaluation"))
  }

  pred <- predict(model, newdata)
  .check_truth_length(truth, NROW(pred))
  metrics <- .regression_metrics(pred, truth)
  structure(list(metrics = metrics, predictions = pred),
            class = "neuralnetwork_evaluation")
}

nn_tune <- function(x,
                    y = NULL,
                    data = NULL,
                    grid = NULL,
                    validation_split = 0.2,
                    metric = c("auto", "loss", "accuracy", "balanced_accuracy",
                               "log_loss", "f1", "rmse", "mae", "rsq"),
                    maximize = NULL,
                    error_action = c("stop", "continue"),
                    seed = NULL,
                    verbose = FALSE,
                    ...) {
  metric <- match.arg(metric)
  error_action <- match.arg(error_action)
  .validate_seed(seed)
  if (is.null(grid)) {
    grid <- list(
      hidden = list(c(16), c(32), c(32, 16)),
      learning_rate = c(0.01, 0.003),
      l2 = c(0, 1e-4),
      dropout = c(0, 0.1)
    )
  }
  candidates <- .expand_tune_grid(grid)
  if (nrow(candidates) == 0L) {
    stop("`grid` produced no tuning candidates.", call. = FALSE)
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }

  models <- vector("list", nrow(candidates))
  scores <- rep(NA_real_, nrow(candidates))
  metric_names <- rep(NA_character_, nrow(candidates))
  errors <- rep(NA_character_, nrow(candidates))

  for (i in seq_len(nrow(candidates))) {
    args <- c(
      list(x = x, y = y, data = data, validation_split = validation_split,
           verbose = verbose),
      .candidate_args(candidates, i),
      list(...)
    )
    if (!is.null(seed)) {
      args$seed <- .offset_seed(seed, i - 1L)
    }
    model <- tryCatch(
      do.call(nn_fit, args),
      error = function(e) {
        if (error_action == "stop") {
          stop(conditionMessage(e), call. = FALSE)
        }
        errors[i] <<- conditionMessage(e)
        NULL
      }
    )
    if (is.null(model)) {
      next
    }
    score <- tryCatch(
      .tune_score(model, metric),
      error = function(e) {
        if (error_action == "stop") {
          stop(conditionMessage(e), call. = FALSE)
        }
        errors[i] <<- conditionMessage(e)
        NULL
      }
    )
    if (is.null(score)) {
      models[[i]] <- model
      next
    }
    models[[i]] <- model
    scores[i] <- score$value
    metric_names[i] <- score$name
  }

  unavailable <- is.na(errors) & is.na(scores)
  errors[unavailable] <- "Candidate produced an unavailable score."
  valid <- which(!is.na(scores))
  if (!length(valid)) {
    first_error <- errors[!is.na(errors)][1L]
    detail <- if (length(first_error) && !is.na(first_error)) {
      paste0(" First error: ", first_error)
    } else {
      ""
    }
    stop("All tuning candidates failed or produced unavailable scores.",
         detail, call. = FALSE)
  }

  if (is.null(maximize)) {
    maximize <- metric_names[valid[1L]] %in% c("accuracy", "balanced_accuracy",
                                               "macro_f1", "f1", "rsq")
  }
  best <- if (isTRUE(maximize)) {
    valid[which.max(scores[valid])]
  } else {
    valid[which.min(scores[valid])]
  }

  results <- candidates
  results$success <- is.na(errors)
  results$metric <- metric_names
  results$score <- scores
  results$error <- errors
  results$rank <- rank(if (isTRUE(maximize)) -scores else scores,
                       ties.method = "first", na.last = "keep")
  results <- results[order(results$rank, na.last = TRUE), , drop = FALSE]

  structure(list(
    results = results,
    best_model = models[[best]],
    best_params = candidates[best, , drop = FALSE],
    models = models,
    errors = errors,
    maximize = maximize
  ), class = "neuralnetwork_tune")
}

nn_cv <- function(x,
                  y = NULL,
                  data = NULL,
                  k = 5,
                  repeats = 1,
                  stratify = TRUE,
                  metric = c("auto", "loss", "accuracy", "balanced_accuracy",
                             "log_loss", "f1", "rmse", "mae", "rsq"),
                  seed = NULL,
                  verbose = FALSE,
                  ...) {
  metric <- match.arg(metric)
  .validate_seed(seed)
  source <- .cv_source(x, y, data, parent.frame())
  folds <- .make_cv_folds(source$y, k, repeats, stratify, seed)
  results <- vector("list", length(folds))
  models <- vector("list", length(folds))

  for (i in seq_along(folds)) {
    fold <- folds[[i]]
    test_idx <- fold$test
    train_idx <- setdiff(seq_len(source$n), test_idx)
    args <- c(.cv_fit_args(source, train_idx),
              list(verbose = verbose),
              list(...))
    if (!is.null(seed)) {
      args$seed <- .offset_seed(seed, i - 1L)
    }
    model <- do.call(nn_fit, args)
    evaluation <- do.call(nn_evaluate, .cv_eval_args(source, model, test_idx))
    score <- .evaluation_score(evaluation, model, metric)
    models[[i]] <- model
    results[[i]] <- data.frame(
      repeat_id = fold$repeat_id,
      fold = fold$fold,
      n = length(test_idx),
      metric = score$name,
      score = score$value,
      stringsAsFactors = FALSE
    )
  }

  results <- do.call(rbind, results)
  summary <- stats::aggregate(score ~ metric, results, function(z) {
    c(mean = mean(z), sd = stats::sd(z))
  })
  summary <- data.frame(
    metric = summary$metric,
    mean = summary$score[, "mean"],
    sd = summary$score[, "sd"],
    row.names = NULL
  )
  structure(list(
    results = results,
    summary = summary,
    models = models,
    k = k,
    repeats = repeats,
    call = match.call()
  ), class = "neuralnetwork_cv")
}

nn_permutation_importance <- function(model,
                                      newdata,
                                      y = NULL,
                                      metric = c("auto", "loss", "accuracy",
                                                 "balanced_accuracy",
                                                 "log_loss", "f1", "rmse",
                                                 "mae", "rsq"),
                                      n_repeats = 5,
                                      seed = NULL) {
  if (!inherits(model, "neuralnetwork")) {
    stop("`model` must be a neuralnetwork object.", call. = FALSE)
  }
  metric <- match.arg(metric)
  if (missing(newdata)) {
    stop("`newdata` is required.", call. = FALSE)
  }
  .validate_seed(seed)
  if (!is.numeric(n_repeats) || length(n_repeats) != 1L ||
      is.na(n_repeats) || !is.finite(n_repeats) || n_repeats < 1 ||
      n_repeats != floor(n_repeats) || n_repeats > .Machine$integer.max) {
    stop("`n_repeats` must be a positive integer.", call. = FALSE)
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }

  truth <- .extract_truth(model, newdata, y)
  x <- .prepare_new_x(newdata, model$blueprint)
  .check_truth_length(truth, nrow(x))
  x <- .apply_scaler(x, model$scaler)
  baseline <- .score_model_matrix(model, x, truth, metric)
  higher_is_better <- baseline$name %in% c("accuracy", "balanced_accuracy",
                                           "macro_f1", "f1", "rsq")
  feature_names <- colnames(x)
  rows <- vector("list", length(feature_names))

  for (j in seq_along(feature_names)) {
    scores <- numeric(n_repeats)
    for (r in seq_len(n_repeats)) {
      xp <- x
      xp[, j] <- sample(xp[, j])
      scores[r] <- .score_model_matrix(model, xp, truth, metric)$value
    }
    permuted <- mean(scores)
    importance <- if (higher_is_better) {
      baseline$value - permuted
    } else {
      permuted - baseline$value
    }
    rows[[j]] <- data.frame(
      feature = feature_names[j],
      importance = importance,
      baseline = baseline$value,
      permuted = permuted,
      metric = baseline$name,
      n_repeats = n_repeats,
      stringsAsFactors = FALSE
    )
  }

  out <- do.call(rbind, rows)
  out <- out[order(out$importance, decreasing = TRUE), , drop = FALSE]
  row.names(out) <- NULL
  structure(out, class = c("neuralnetwork_importance", "data.frame"))
}

print.neuralnetwork_evaluation <- function(x, ...) {
  cat("neuralnetwork evaluation\n")
  cat("Metrics:\n")
  print(.metric_table(x$metrics), row.names = FALSE)
  if (!is.null(x$confusion)) {
    cat("\nConfusion matrix:\n")
    print(x$confusion)
  }
  invisible(x)
}

print.neuralnetwork_tune <- function(x, ...) {
  top <- utils::head(x$results, 5L)
  direction <- if (isTRUE(x$maximize)) "higher is better" else "lower is better"
  failed <- if ("success" %in% names(x$results)) {
    sum(!x$results$success)
  } else {
    0L
  }
  cat("neuralnetwork tuning result\n")
  cat("  Candidates: ", nrow(x$results), "\n", sep = "")
  if (failed > 0L) {
    cat("  Failed:     ", failed, "\n", sep = "")
  }
  cat("  Objective:  ", top$metric[1L], " (", direction, ")\n", sep = "")
  cat("  Best score: ", .fmt(top$score[1L]), "\n\n", sep = "")
  cat("Top candidates:\n")
  print(.format_score_frame(top), row.names = FALSE)
  invisible(x)
}

print.neuralnetwork_cv <- function(x, ...) {
  cat("neuralnetwork cross-validation\n")
  cat("  Folds:   ", x$k, "\n", sep = "")
  cat("  Repeats: ", x$repeats, "\n\n", sep = "")
  print(.format_score_frame(x$summary), row.names = FALSE)
  invisible(x)
}

print.neuralnetwork_importance <- function(x, ...) {
  shown <- utils::head(x, 10L)
  cat("neuralnetwork permutation importance\n")
  cat("  Metric:  ", shown$metric[1L], "\n", sep = "")
  cat("  Repeats: ", shown$n_repeats[1L], "\n\n", sep = "")
  print.data.frame(.format_score_frame(shown), row.names = FALSE)
  invisible(x)
}

.task_label <- function(task) {
  if (identical(task, "binary_classification")) {
    "binary classification"
  } else {
    task
  }
}

.loss_label <- function(model) {
  loss <- model$loss %||% if (model$task == "regression") {
    "squared_error"
  } else {
    "cross_entropy"
  }
  if (identical(loss, "huber")) {
    paste0("huber (delta=", .fmt(model$huber_delta %||% 1), ")")
  } else {
    loss
  }
}

.final_history <- function(model) {
  model$history[nrow(model$history), , drop = FALSE]
}

.fmt <- function(x, digits = 5) {
  if (length(x) == 0L || is.na(x)) {
    return("NA")
  }
  format(signif(as.numeric(x), digits), trim = TRUE, scientific = FALSE)
}

.metric_table <- function(metrics, digits = 5) {
  data.frame(
    metric = names(metrics),
    value = vapply(metrics, .fmt, character(1), digits = digits),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

.format_history_row <- function(x) {
  out <- x
  numeric_cols <- vapply(out, is.numeric, logical(1))
  out[numeric_cols] <- lapply(out[numeric_cols], function(z) {
    vapply(z, .fmt, character(1))
  })
  out
}

.format_score_frame <- function(x, digits = 5) {
  out <- x
  numeric_cols <- vapply(out, is.numeric, logical(1))
  out[numeric_cols] <- lapply(out[numeric_cols], function(z) {
    vapply(z, .fmt, character(1), digits = digits)
  })
  list_cols <- vapply(out, is.list, logical(1))
  out[list_cols] <- lapply(out[list_cols], function(z) {
    vapply(z, function(value) {
      if (length(value) == 0L) {
        ""
      } else {
        paste(unlist(value), collapse = ", ")
      }
    }, character(1))
  })
  out
}

.new_history <- function(n) {
  data.frame(
    epoch = integer(n),
    train_loss = numeric(n),
    validation_loss = numeric(n),
    train_metric = numeric(n),
    validation_metric = numeric(n),
    gradient_norm = numeric(n),
    learning_rate = numeric(n),
    backtracked = logical(n)
  )
}

.prepare_xy <- function(x, y, data) {
  if (inherits(x, "formula")) {
    if (is.null(data) && is.data.frame(y)) {
      data <- y
    }
    return(.prepare_formula_xy(x, data))
  }

  if (is.null(y)) {
    stop("`y` is required when `x` is not a formula.", call. = FALSE)
  }

  x_prepared <- .prepare_x_matrix(x)
  list(x = x_prepared$x, y = y, blueprint = x_prepared$blueprint)
}

.prepare_formula_xy <- function(formula, data) {
  if (is.null(data)) {
    data <- parent.frame()
  }

  mf <- stats::model.frame(formula, data = data, na.action = stats::na.fail)
  terms_obj <- stats::terms(formula, data = data)
  x_terms <- stats::delete.response(terms_obj)
  y <- stats::model.response(mf)
  x <- stats::model.matrix(x_terms, mf)
  x <- .drop_intercept(x)

  list(
    x = .double_matrix(x),
    y = y,
    blueprint = list(
      kind = "formula",
      terms = terms_obj,
      x_terms = x_terms,
      contrasts = attr(x, "contrasts"),
      xlevels = stats::.getXlevels(terms_obj, mf),
      feature_names = colnames(x)
    )
  )
}

.prepare_x_matrix <- function(x) {
  if (is.data.frame(x)) {
    terms_obj <- stats::terms(~ ., data = x)
    mf <- stats::model.frame(terms_obj, data = x, na.action = stats::na.fail)
    mat <- stats::model.matrix(terms_obj, mf)
    mat <- .drop_intercept(mat)
    return(list(
      x = .double_matrix(mat),
      blueprint = list(
        kind = "data.frame",
        terms = terms_obj,
        x_terms = terms_obj,
        contrasts = attr(mat, "contrasts"),
        xlevels = stats::.getXlevels(terms_obj, mf),
        feature_names = colnames(mat)
      )
    ))
  }

  mat <- .double_matrix(x)
  if (is.null(colnames(mat))) {
    colnames(mat) <- paste0("x", seq_len(ncol(mat)))
  }

  list(
    x = mat,
    blueprint = list(
      kind = "matrix",
      feature_names = colnames(mat),
      n_features = ncol(mat)
    )
  )
}

.prepare_new_x <- function(newdata, blueprint) {
  if (blueprint$kind %in% c("formula", "data.frame")) {
    mf <- tryCatch(
      stats::model.frame(blueprint$x_terms, data = newdata,
                         na.action = stats::na.pass,
                         xlev = blueprint$xlevels),
      error = function(e) {
        message <- conditionMessage(e)
        if (grepl("new level", message, fixed = TRUE) ||
            grepl("new levels", message, fixed = TRUE)) {
          stop("`newdata` contains factor levels not seen during fitting. ",
               message, call. = FALSE)
        }
        stop("Could not prepare `newdata` with the fitted predictors. ",
             message, call. = FALSE)
      }
    )
    mat <- stats::model.matrix(blueprint$x_terms, mf,
                               contrasts.arg = blueprint$contrasts)
    mat <- .drop_intercept(mat)
    missing_cols <- setdiff(blueprint$feature_names, colnames(mat))
    if (length(missing_cols) > 0L) {
      zeros <- matrix(0, nrow = nrow(mat), ncol = length(missing_cols))
      colnames(zeros) <- missing_cols
      mat <- cbind(mat, zeros)
    }
    mat <- mat[, blueprint$feature_names, drop = FALSE]
    return(.double_matrix(mat))
  }

  mat <- .double_matrix(newdata)
  if (!is.null(colnames(mat)) &&
      all(blueprint$feature_names %in% colnames(mat))) {
    mat <- mat[, blueprint$feature_names, drop = FALSE]
  } else if (ncol(mat) != blueprint$n_features) {
    stop("`newdata` has ", ncol(mat), " columns; expected ",
         blueprint$n_features, ".", call. = FALSE)
  }
  colnames(mat) <- blueprint$feature_names
  mat
}

.encode_y <- function(y, task) {
  if (anyNA(y)) {
    stop("Outcome values cannot contain missing values.", call. = FALSE)
  }

  if (task == "auto") {
    task <- if (is.factor(y) || is.character(y) || is.logical(y)) {
      "classification"
    } else {
      "regression"
    }
  }

  if (task == "classification") {
    if (is.matrix(y) || is.data.frame(y)) {
      mat <- .double_matrix(y)
      classes <- colnames(mat)
      if (is.null(classes)) {
        classes <- paste0("class", seq_len(ncol(mat)))
      }
      if (ncol(mat) == 1L) {
        if (any(mat < 0 | mat > 1)) {
          stop("One-column classification matrices must contain probabilities or 0/1 labels.",
               call. = FALSE)
        }
        class_index <- ifelse(mat[, 1L] >= 0.5, 2L, 1L)
        return(list(
          y = mat,
          task = "binary_classification",
          classes = c("0", "1"),
          output_names = classes,
          class_index = class_index
        ))
      }
      mat <- .validate_class_matrix(mat)
      class_index <- max.col(mat, ties.method = "first")
      return(list(
        y = mat,
        task = "classification",
        classes = classes,
        output_names = classes,
        class_index = class_index
      ))
    }

    y_factor <- factor(y)
    classes <- levels(y_factor)
    if (length(classes) < 2L) {
      stop("Classification requires at least two outcome classes.", call. = FALSE)
    }
    if (length(classes) == 2L) {
      mat <- matrix(as.integer(y_factor) - 1L, ncol = 1L)
      colnames(mat) <- classes[2L]
      return(list(
        y = mat,
        task = "binary_classification",
        classes = classes,
        output_names = classes[2L],
        class_index = as.integer(y_factor)
      ))
    }
    mat <- matrix(0, nrow = length(y_factor), ncol = length(classes))
    mat[cbind(seq_along(y_factor), as.integer(y_factor))] <- 1
    colnames(mat) <- classes
    return(list(
      y = mat,
      task = "classification",
      classes = classes,
      output_names = classes,
      class_index = as.integer(y_factor)
    ))
  }

  mat <- .double_matrix(y)
  if (is.null(colnames(mat))) {
    colnames(mat) <- if (ncol(mat) == 1L) "y" else paste0("y", seq_len(ncol(mat)))
  }
  list(
    y = mat,
    task = "regression",
    classes = NULL,
    output_names = colnames(mat),
    class_index = NULL
  )
}

.validate_class_matrix <- function(mat) {
  if (any(mat < 0 | mat > 1)) {
    stop("Classification matrices must contain class indicators or probabilities in [0, 1].",
         call. = FALSE)
  }
  row_total <- rowSums(mat)
  if (any(abs(row_total - 1) > 1e-8)) {
    stop("Rows of a classification matrix must sum to 1.",
         call. = FALSE)
  }
  mat
}

.double_matrix <- function(x) {
  if (is.vector(x) && !is.list(x)) {
    x <- matrix(x, ncol = 1L)
  } else {
    x <- as.matrix(x)
  }
  storage.mode(x) <- "double"
  if (anyNA(x)) {
    stop("Input data contains missing values. Impute or remove them before fitting or prediction.",
         call. = FALSE)
  }
  if (any(!is.finite(x))) {
    stop("Non-finite values are not supported.", call. = FALSE)
  }
  x
}

.fit_target_scaler <- function(y, task, y_scale) {
  if (task != "regression" || !isTRUE(y_scale)) {
    return(list(enabled = FALSE, center = rep(0, ncol(y)), scale = rep(1, ncol(y))))
  }
  center <- colMeans(y)
  spread <- apply(y, 2L, stats::sd)
  spread[is.na(spread) | spread == 0] <- 1
  list(enabled = TRUE, center = center, scale = spread)
}

.apply_target_scaler <- function(y, target_scaler) {
  if (is.null(target_scaler) || !isTRUE(target_scaler$enabled)) {
    return(y)
  }
  sweep(sweep(y, 2L, target_scaler$center, "-"), 2L, target_scaler$scale, "/")
}

.inverse_target_scaler <- function(y, target_scaler) {
  if (is.null(target_scaler) || !isTRUE(target_scaler$enabled)) {
    return(y)
  }
  sweep(sweep(y, 2L, target_scaler$scale, "*"), 2L, target_scaler$center, "+")
}

.make_sample_weights <- function(encoded, sample_weight, class_weight) {
  n <- nrow(encoded$y)
  weights <- if (is.null(sample_weight)) {
    rep(1, n)
  } else {
    if (!is.numeric(sample_weight) || length(sample_weight) != n ||
        anyNA(sample_weight) || any(!is.finite(sample_weight)) ||
        any(sample_weight < 0)) {
      stop("`sample_weight` must be a non-negative finite numeric vector with one value per row.",
           call. = FALSE)
    }
    as.numeric(sample_weight)
  }

  if (!is.null(class_weight)) {
    if (is.null(encoded$class_index)) {
      stop("`class_weight` is only available for classification tasks.",
           call. = FALSE)
    }
    class_weights <- .resolve_class_weight(class_weight, encoded)
    weights <- weights * class_weights[encoded$class_index]
  }

  if (sum(weights) <= 0) {
    stop("Combined sample weights must have a positive sum.", call. = FALSE)
  }

  weights
}

.resolve_class_weight <- function(class_weight, encoded) {
  classes <- encoded$classes
  if (is.character(class_weight) && length(class_weight) == 1L &&
      identical(class_weight, "balanced")) {
    counts <- tabulate(encoded$class_index, nbins = length(classes))
    counts[counts == 0] <- 1
    return(sum(counts) / (length(classes) * counts))
  }

  if (!is.numeric(class_weight) || anyNA(class_weight) ||
      any(!is.finite(class_weight)) || any(class_weight < 0)) {
    stop("`class_weight` must be \"balanced\" or a non-negative numeric vector.",
         call. = FALSE)
  }

  if (!is.null(names(class_weight))) {
    missing <- setdiff(classes, names(class_weight))
    if (length(missing) > 0L) {
      stop("`class_weight` is missing weights for: ",
           paste(missing, collapse = ", "), call. = FALSE)
    }
    return(as.numeric(class_weight[classes]))
  }

  if (length(class_weight) != length(classes)) {
    stop("Unnamed `class_weight` must have one value per class.", call. = FALSE)
  }

  as.numeric(class_weight)
}

.drop_intercept <- function(x) {
  intercept <- which(colnames(x) == "(Intercept)")
  if (length(intercept) > 0L) {
    x <- x[, -intercept, drop = FALSE]
  }
  x
}

.fit_scaler <- function(x, scale) {
  center <- if (isTRUE(scale)) colMeans(x) else rep(0, ncol(x))
  spread <- if (isTRUE(scale)) {
    apply(x, 2L, stats::sd)
  } else {
    rep(1, ncol(x))
  }
  spread[is.na(spread) | spread == 0] <- 1
  list(center = center, scale = spread)
}

.apply_scaler <- function(x, scaler) {
  if (nrow(x) == 0L) {
    return(x)
  }
  sweep(sweep(x, 2L, scaler$center, "-"), 2L, scaler$scale, "/")
}

.make_validation_split <- function(n, validation_split) {
  if (validation_split <= 0) {
    return(list(train = seq_len(n), validation = integer()))
  }
  n_val <- max(1L, floor(n * validation_split))
  if (n_val >= n) {
    stop("`validation_split` leaves no training rows.", call. = FALSE)
  }
  validation <- sort(sample.int(n, n_val))
  train <- setdiff(seq_len(n), validation)
  list(train = train, validation = validation)
}

.validate_hidden <- function(hidden) {
  if (is.character(hidden) && length(hidden) == 1L &&
      identical(hidden, "auto")) {
    return("auto")
  }
  if (length(hidden) == 0L) {
    return(integer(0))
  }
  if (is.character(hidden)) {
    stop("`hidden` must contain positive integer layer sizes, or 0 for no hidden layer.",
         call. = FALSE)
  }
  if (length(hidden) == 1L && identical(as.numeric(hidden), 0)) {
    return(integer(0))
  }
  if (!is.numeric(hidden) || anyNA(hidden) || any(!is.finite(hidden)) ||
      any(hidden < 1) || any(hidden != floor(hidden))) {
    stop("`hidden` must contain positive integer layer sizes, or 0 for no hidden layer.",
         call. = FALSE)
  }
  as.integer(hidden)
}

.resolve_hidden <- function(hidden, task, x, output_dim) {
  if (!identical(hidden, "auto")) {
    return(hidden)
  }
  p <- ncol(x)
  if (task == "regression") {
    return(min(12L, max(6L, p + 3L)))
  }
  if (task == "binary_classification") {
    return(min(12L, max(8L, p * 2L)))
  }
  min(12L, max(4L, p))
}

.validate_dropout <- function(dropout, n_hidden) {
  if (!is.numeric(dropout) || anyNA(dropout) || any(!is.finite(dropout))) {
    stop("`dropout` must be numeric values in [0, 1).", call. = FALSE)
  }
  if (n_hidden == 0L) {
    if (length(dropout) == 1L && dropout == 0) {
      return(numeric(0))
    }
    stop("`dropout` must be 0 when `hidden = 0`.", call. = FALSE)
  }
  if (length(dropout) == 1L) {
    dropout <- rep(dropout, n_hidden)
  }
  if (length(dropout) != n_hidden || any(dropout < 0 | dropout >= 1)) {
    stop("`dropout` must be in [0, 1) and match the number of hidden layers.",
         call. = FALSE)
  }
  dropout
}

.resolve_epochs <- function(epochs, stepmax) {
  if (!is.null(stepmax)) {
    if (!is.numeric(stepmax) || length(stepmax) != 1L || is.na(stepmax) ||
        !is.finite(stepmax) || stepmax < 1 || stepmax != floor(stepmax) ||
        stepmax > .Machine$integer.max) {
      stop("`stepmax` must be a positive integer when supplied.",
           call. = FALSE)
    }
    return(as.integer(stepmax))
  }
  epochs
}

.validate_training_args <- function(epochs, batch_size, learning_rate, l2,
                                    validation_split, gradient_clip,
                                    learning_rate_decay, threshold,
                                    momentum, beta1, beta2, epsilon,
                                    patience, min_delta) {
  if (!is.numeric(epochs) || length(epochs) != 1L || is.na(epochs) ||
      !is.finite(epochs) || epochs < 1 || epochs != floor(epochs) ||
      epochs > .Machine$integer.max) {
    stop("`epochs` must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(batch_size) || length(batch_size) != 1L || is.na(batch_size) ||
      !is.finite(batch_size) || batch_size < 1 ||
      batch_size != floor(batch_size) || batch_size > .Machine$integer.max) {
    stop("`batch_size` must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(learning_rate) || length(learning_rate) != 1L ||
      is.na(learning_rate) || !is.finite(learning_rate) || learning_rate <= 0) {
    stop("`learning_rate` must be a positive finite number.", call. = FALSE)
  }
  if (!is.numeric(l2) || length(l2) != 1L || is.na(l2) ||
      !is.finite(l2) || l2 < 0) {
    stop("`l2` must be a finite non-negative number.", call. = FALSE)
  }
  if (!is.numeric(validation_split) || length(validation_split) != 1L ||
      is.na(validation_split) || !is.finite(validation_split) ||
      validation_split < 0 || validation_split >= 1) {
    stop("`validation_split` must be in [0, 1).", call. = FALSE)
  }
  if (!is.numeric(gradient_clip) || length(gradient_clip) != 1L ||
      is.na(gradient_clip) || gradient_clip <= 0) {
    stop("`gradient_clip` must be a positive number or Inf.", call. = FALSE)
  }
  if (!is.numeric(learning_rate_decay) || length(learning_rate_decay) != 1L ||
      is.na(learning_rate_decay) || !is.finite(learning_rate_decay) ||
      learning_rate_decay < 0) {
    stop("`learning_rate_decay` must be a finite non-negative number.",
         call. = FALSE)
  }
  if (!is.numeric(threshold) || length(threshold) != 1L ||
      is.na(threshold) || !is.finite(threshold) || threshold < 0) {
    stop("`threshold` must be a finite non-negative number.", call. = FALSE)
  }
  if (!is.numeric(momentum) || length(momentum) != 1L ||
      is.na(momentum) || !is.finite(momentum) ||
      momentum < 0 || momentum >= 1) {
    stop("`momentum` must be in [0, 1).", call. = FALSE)
  }
  if (!is.numeric(beta1) || length(beta1) != 1L || is.na(beta1) ||
      !is.finite(beta1) || beta1 < 0 || beta1 >= 1) {
    stop("`beta1` must be in [0, 1).", call. = FALSE)
  }
  if (!is.numeric(beta2) || length(beta2) != 1L || is.na(beta2) ||
      !is.finite(beta2) || beta2 < 0 || beta2 >= 1) {
    stop("`beta2` must be in [0, 1).", call. = FALSE)
  }
  if (!is.numeric(epsilon) || length(epsilon) != 1L || is.na(epsilon) ||
      !is.finite(epsilon) || epsilon <= 0) {
    stop("`epsilon` must be a positive finite number.", call. = FALSE)
  }
  patience_ok <- is.numeric(patience) && length(patience) == 1L &&
    !is.na(patience) &&
    ((is.infinite(patience) && patience > 0) ||
      (is.finite(patience) && patience >= 1 &&
        patience == floor(patience) && patience <= .Machine$integer.max))
  if (!patience_ok) {
    stop("`patience` must be a positive integer or Inf.", call. = FALSE)
  }
  if (!is.numeric(min_delta) || length(min_delta) != 1L ||
      is.na(min_delta) || !is.finite(min_delta) || min_delta < 0) {
    stop("`min_delta` must be a finite non-negative number.", call. = FALSE)
  }
  invisible(TRUE)
}

.resolve_loss <- function(loss, task, huber_delta) {
  if (!is.numeric(huber_delta) || length(huber_delta) != 1L ||
      is.na(huber_delta) || !is.finite(huber_delta) || huber_delta <= 0) {
    stop("`huber_delta` must be a positive finite number.", call. = FALSE)
  }
  if (task %in% c("classification", "binary_classification")) {
    if (loss != "auto") {
      stop("`loss` is only configurable for regression; use `loss = \"auto\"` for classification.",
           call. = FALSE)
    }
    return("cross_entropy")
  }
  if (loss == "auto") {
    return("squared_error")
  }
  loss
}

.extract_truth <- function(model, newdata, y) {
  if (!is.null(y)) {
    return(y)
  }
  if (model$blueprint$kind == "formula") {
    response <- all.vars(model$blueprint$terms)[1L]
    if (is.data.frame(newdata) && response %in% names(newdata)) {
      return(newdata[[response]])
    }
  }
  stop("Provide `y` or pass `newdata` with the original response column.",
       call. = FALSE)
}

.check_truth_length <- function(truth, n) {
  if (NROW(truth) != n) {
    stop("Outcome data must have one row per prediction row.",
         call. = FALSE)
  }
  invisible(TRUE)
}

.validate_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop("`", name, "` must be TRUE or FALSE.", call. = FALSE)
  }
  invisible(TRUE)
}

.validate_seed <- function(seed) {
  if (is.null(seed)) {
    return(invisible(TRUE))
  }
  if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) ||
      !is.finite(seed) || seed < 0 || seed != floor(seed) ||
      seed > .Machine$integer.max) {
    stop("`seed` must be NULL or a non-negative integer.", call. = FALSE)
  }
  invisible(TRUE)
}

.offset_seed <- function(seed, offset) {
  if (is.null(seed)) {
    return(NULL)
  }
  integer_limit <- as.numeric(.Machine$integer.max)
  seed_value <- as.numeric(seed) + as.numeric(offset)
  while (seed_value > integer_limit) {
    seed_value <- seed_value - integer_limit
  }
  as.integer(seed_value)
}

.validate_positive_scalar <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) ||
      !is.finite(x) || x <= 0) {
    stop("`", name, "` must be a positive finite number.", call. = FALSE)
  }
  invisible(TRUE)
}

.validate_positive_integer <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) ||
      !is.finite(x) || x < 1 || x != floor(x)) {
    stop("`", name, "` must be a positive integer.", call. = FALSE)
  }
  invisible(TRUE)
}

.validate_probability_level <- function(level) {
  if (!is.numeric(level) || length(level) != 1L || is.na(level) ||
      !is.finite(level) || level <= 0 || level >= 1) {
    stop("`level` must be a number between 0 and 1.", call. = FALSE)
  }
  invisible(TRUE)
}

.safe_divide <- function(num, den) {
  out <- rep(NA_real_, length(num))
  ok <- den != 0 & is.finite(den)
  out[ok] <- num[ok] / den[ok]
  out
}

.classification_metrics <- function(truth, pred, prob, classes) {
  truth <- factor(truth, levels = classes)
  pred <- factor(pred, levels = classes)
  truth_index <- as.integer(truth)
  pred_index <- as.integer(pred)
  if (anyNA(truth_index)) {
    stop("Truth contains classes not seen during fitting.", call. = FALSE)
  }
  .classification_metrics_from_indices(truth_index, pred_index, prob, classes)
}

.classification_metrics_from_indices <- function(truth_index, pred_index,
                                                 prob, classes) {
  k <- length(classes)
  confusion <- table(
    truth = factor(truth_index, levels = seq_len(k)),
    estimate = factor(pred_index, levels = seq_len(k))
  )
  diag_values <- diag(confusion)
  row_totals <- rowSums(confusion)
  col_totals <- colSums(confusion)
  recall_by_class <- .safe_divide(diag_values, row_totals)
  precision_by_class <- .safe_divide(diag_values, col_totals)
  f1_by_class <- .safe_divide(2 * precision_by_class * recall_by_class,
                              precision_by_class + recall_by_class)
  idx <- cbind(seq_along(truth_index), truth_index)
  metrics <- c(
    accuracy = mean(pred_index == truth_index),
    balanced_accuracy = mean(recall_by_class, na.rm = TRUE),
    macro_precision = mean(precision_by_class, na.rm = TRUE),
    macro_recall = mean(recall_by_class, na.rm = TRUE),
    macro_f1 = mean(f1_by_class, na.rm = TRUE),
    log_loss = -mean(log(pmax(prob[idx], 1e-15)))
  )
  if (k == 2L) {
    precision <- precision_by_class[2L]
    recall <- recall_by_class[2L]
    metrics <- c(
      metrics,
      sensitivity = recall,
      specificity = recall_by_class[1L],
      precision = precision,
      recall = recall,
      f1 = .safe_divide(2 * precision * recall, precision + recall)
    )
  }
  metrics
}

.regression_metrics <- function(pred, truth) {
  truth <- .double_matrix(truth)
  if (ncol(truth) == 1L) {
    truth_values <- as.numeric(truth[, 1L])
    err <- pred - truth_values
  } else {
    truth_values <- truth
    err <- pred - truth_values
  }
  sse <- sum(err^2)
  sst <- sum((truth - colMeans(truth))^2)
  c(
    rmse = sqrt(mean(err^2)),
    mae = mean(abs(err)),
    rsq = if (sst == 0) NA_real_ else 1 - sse / sst
  )
}

.prob_from_output <- function(out, model) {
  if (model$task == "binary_classification") {
    prob <- cbind(1 - out[, 1L], out[, 1L])
    colnames(prob) <- model$classes
    return(prob)
  }
  if (model$task == "classification") {
    colnames(out) <- model$classes
    return(out)
  }
  NULL
}

.metric_from_score_set <- function(metrics, metric) {
  if (metric == "auto") {
    metric <- if ("accuracy" %in% names(metrics)) "accuracy" else "rmse"
  } else if (metric == "loss") {
    metric <- if ("log_loss" %in% names(metrics)) "log_loss" else "rmse"
  } else if (metric == "f1" && !"f1" %in% names(metrics) &&
             "macro_f1" %in% names(metrics)) {
    metric <- "macro_f1"
  }
  if (!metric %in% names(metrics)) {
    stop("Metric `", metric, "` is not available for this task.",
         call. = FALSE)
  }
  list(name = metric, value = unname(metrics[[metric]]))
}

.score_output_metric <- function(out, y_raw, model, metric) {
  if (model$task %in% c("classification", "binary_classification")) {
    prob <- .prob_from_output(out, model)
    truth_index <- if (model$task == "binary_classification") {
      as.integer(y_raw[, 1L] >= 0.5) + 1L
    } else {
      max.col(y_raw, ties.method = "first")
    }
    pred_index <- max.col(prob, ties.method = "first")
    metrics <- .classification_metrics_from_indices(truth_index, pred_index,
                                                    prob, model$classes)
    return(.metric_from_score_set(metrics, metric))
  }
  pred <- .inverse_target_scaler(out, model$target_scaler %||% NULL)
  metrics <- .regression_metrics(pred, y_raw)
  .metric_from_score_set(metrics, metric)
}

.score_model_validation <- function(model, metric) {
  if (is.null(model$validation) || nrow(model$validation$x) == 0L) {
    return(NULL)
  }
  out <- .forward(model$validation$x, model$params, model$activation,
                  model$task, dropout = 0, training = FALSE,
                  backend = model$backend %||% "r")$output
  if (metric == "loss") {
    return(list(
      name = "loss",
      value = .loss_from_output(
        out,
        model$validation$y,
        model$task,
        loss = model$loss %||% "squared_error",
        huber_delta = model$huber_delta %||% 1
      )
    ))
  }
  .score_output_metric(out, model$validation$y_raw, model, metric)
}

.cv_source <- function(x, y, data, env) {
  if (inherits(x, "formula")) {
    if (is.null(data) && is.data.frame(y)) {
      data <- y
    }
    if (is.null(data)) {
      data <- stats::model.frame(x, data = env, na.action = stats::na.fail)
    }
    response <- all.vars(x)[1L]
    if (!is.data.frame(data) || !response %in% names(data)) {
      stop("Formula cross-validation requires `data` with the response column.",
           call. = FALSE)
    }
    return(list(kind = "formula", formula = x, data = data,
                y = data[[response]], n = nrow(data)))
  }

  if (is.null(y)) {
    stop("`y` is required when `x` is not a formula.", call. = FALSE)
  }
  n <- NROW(y)
  if (NROW(x) != n) {
    stop("`x` and `y` must have the same number of rows.", call. = FALSE)
  }
  list(kind = "xy", x = x, y = y, n = n)
}

.subset_rows <- function(x, rows) {
  if (is.null(dim(x))) {
    x[rows]
  } else {
    x[rows, , drop = FALSE]
  }
}

.cv_fit_args <- function(source, rows) {
  if (source$kind == "formula") {
    return(list(x = source$formula,
                data = source$data[rows, , drop = FALSE]))
  }
  list(x = .subset_rows(source$x, rows),
       y = .subset_rows(source$y, rows))
}

.cv_eval_args <- function(source, model, rows) {
  if (source$kind == "formula") {
    return(list(model = model,
                newdata = source$data[rows, , drop = FALSE]))
  }
  list(model = model,
       newdata = .subset_rows(source$x, rows),
       y = .subset_rows(source$y, rows))
}

.make_cv_folds <- function(y, k, repeats, stratify, seed) {
  n <- NROW(y)
  if (!is.numeric(k) || length(k) != 1L || is.na(k) ||
      !is.finite(k) || k < 2 || k != floor(k) || k > n ||
      k > .Machine$integer.max) {
    stop("`k` must be an integer between 2 and the number of rows.",
         call. = FALSE)
  }
  if (!is.numeric(repeats) || length(repeats) != 1L || is.na(repeats) ||
      !is.finite(repeats) || repeats < 1 || repeats != floor(repeats) ||
      repeats > .Machine$integer.max) {
    stop("`repeats` must be a positive integer.", call. = FALSE)
  }
  if (!is.logical(stratify) || length(stratify) != 1L || is.na(stratify)) {
    stop("`stratify` must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }
  if (anyNA(y)) {
    stop("Outcome values cannot contain missing values.", call. = FALSE)
  }

  folds <- list()
  class_like <- is.factor(y) || is.character(y) || is.logical(y)
  if (isTRUE(stratify) && class_like) {
    class_counts <- table(factor(y))
    if (any(class_counts < k)) {
      stop("`k` cannot exceed the smallest class size when `stratify = TRUE`.",
           call. = FALSE)
    }
  }
  for (rep_id in seq_len(repeats)) {
    fold_id <- integer(n)
    if (isTRUE(stratify) && class_like) {
      split_rows <- split(seq_len(n), factor(y))
      for (idx in split_rows) {
        fold_id[idx] <- sample(rep(seq_len(k), length.out = length(idx)))
      }
    } else {
      fold_id <- sample(rep(seq_len(k), length.out = n))
    }
    for (fold in seq_len(k)) {
      folds[[length(folds) + 1L]] <- list(
        repeat_id = rep_id,
        fold = fold,
        test = which(fold_id == fold)
      )
    }
  }
  folds
}

.evaluation_score <- function(evaluation, model, metric) {
  .metric_from_score_set(evaluation$metrics, metric)
}

.score_model_matrix <- function(model, x, truth, metric) {
  out <- .forward(x, model$params, model$activation, model$task, dropout = 0,
                  training = FALSE, backend = model$backend %||% "r")$output
  if (metric == "loss") {
    truth_encoded <- .encode_truth_for_model(model, truth)
    return(list(name = "loss",
                value = .loss_from_output(
                  out,
                  truth_encoded,
                  model$task,
                  loss = model$loss %||% "squared_error",
                  huber_delta = model$huber_delta %||% 1
                )))
  }
  if (model$task %in% c("classification", "binary_classification")) {
    truth <- factor(truth, levels = model$classes)
    if (anyNA(truth)) {
      stop("Truth contains classes not seen during fitting.", call. = FALSE)
    }
    prob <- .prob_from_output(out, model)
    pred <- factor(model$classes[max.col(prob, ties.method = "first")],
                   levels = model$classes)
    metrics <- .classification_metrics(truth, pred, prob, model$classes)
    return(.metric_from_score_set(metrics, metric))
  } else {
    truth <- .double_matrix(truth)
    pred <- .inverse_target_scaler(out, model$target_scaler %||% NULL)
    metrics <- .regression_metrics(pred, truth)
    return(.metric_from_score_set(metrics, metric))
  }
}

.expand_tune_grid <- function(grid) {
  if (!is.list(grid) || length(grid) == 0L) {
    stop("`grid` must be a named list of candidate values.", call. = FALSE)
  }
  if (is.null(names(grid)) || any(names(grid) == "")) {
    stop("Every `grid` entry must be named.", call. = FALSE)
  }
  grid <- lapply(grid, function(x) {
    if (is.list(x)) x else as.list(x)
  })
  combos <- expand.grid(lapply(grid, seq_along), KEEP.OUT.ATTRS = FALSE)
  names(combos) <- names(grid)
  out <- vector("list", length(grid))
  names(out) <- names(grid)
  for (name in names(grid)) {
    out[[name]] <- I(grid[[name]][combos[[name]]])
  }
  as.data.frame(out, stringsAsFactors = FALSE)
}

.candidate_args <- function(candidates, i) {
  args <- vector("list", ncol(candidates))
  names(args) <- names(candidates)
  for (name in names(candidates)) {
    value <- candidates[[name]][[i]]
    args[[name]] <- value
  }
  args
}

.tune_score <- function(model, metric) {
  history <- model$history
  best_epoch <- model$best_epoch %||% nrow(history)
  best_epoch <- max(1L, best_epoch)
  row <- history[best_epoch, , drop = FALSE]
  if (metric == "auto") {
    metric <- if (model$task %in% c("classification", "binary_classification")) {
      "accuracy"
    } else {
      "rmse"
    }
  }
  if (metric == "loss") {
    value <- if (!is.na(row$validation_loss)) row$validation_loss else row$train_loss
    return(list(name = "loss", value = value))
  }
  validation_score <- .score_model_validation(model, metric)
  if (!is.null(validation_score)) {
    return(validation_score)
  }
  if (metric == "accuracy") {
    value <- if (!is.na(row$validation_metric)) row$validation_metric else row$train_metric
    return(list(name = "accuracy", value = value))
  }
  if (metric == "rmse") {
    value <- if (!is.na(row$validation_metric)) row$validation_metric else row$train_metric
    return(list(name = "rmse", value = value))
  }
  stop("`metric = \"", metric, "\"` requires `validation_split > 0`.",
       call. = FALSE)
}

.validate_callbacks <- function(callbacks) {
  if (is.null(callbacks)) {
    return(list())
  }
  if (is.function(callbacks)) {
    callbacks <- list(callbacks)
  }
  if (!is.list(callbacks) || !all(vapply(callbacks, is.function, logical(1)))) {
    stop("`callbacks` must be a function or a list of functions.", call. = FALSE)
  }
  callbacks
}

.run_callbacks <- function(callbacks, state) {
  result <- list(stop = FALSE, learning_rate = state$learning_rate)
  if (length(callbacks) == 0L) {
    return(result)
  }

  for (callback in callbacks) {
    value <- callback(state)
    if (identical(value, FALSE)) {
      result$stop <- TRUE
    } else if (is.list(value)) {
      if (!is.null(value$stop)) {
        if (!is.logical(value$stop) || length(value$stop) != 1L ||
            is.na(value$stop)) {
          stop("Callback `stop` values must be TRUE or FALSE.",
               call. = FALSE)
        }
        result$stop <- isTRUE(value$stop)
      }
      if (!is.null(value$learning_rate)) {
        if (!is.numeric(value$learning_rate) || length(value$learning_rate) != 1L ||
            is.na(value$learning_rate) || !is.finite(value$learning_rate) ||
            value$learning_rate <= 0) {
          stop("Callback `learning_rate` values must be positive finite scalars.",
               call. = FALSE)
        }
        result$learning_rate <- value$learning_rate
        state$learning_rate <- result$learning_rate
      }
    }
    if (isTRUE(result$stop)) {
      break
    }
  }
  result
}

.resolve_backend <- function(backend) {
  if (backend == "auto") {
    return(if (.has_rcpp_backend()) "rcpp" else "r")
  }
  if (backend == "rcpp" && !.has_rcpp_backend()) {
    stop("The Rcpp backend is not loaded. Reinstall neuralnetwork from source or use `backend = \"r\"`.",
         call. = FALSE)
  }
  backend
}

.resolve_activation <- function(activation, task, x, hidden) {
  if (activation != "auto") {
    return(activation)
  }
  if (length(hidden) == 0L) {
    return("tanh")
  }
  if (task == "regression") {
    return("sigmoid")
  }
  if (task == "binary_classification") {
    return("relu")
  }
  "tanh"
}

.resolve_optimizer <- function(optimizer, task, x, hidden, dropout, output_dim,
                               callbacks = NULL) {
  if (optimizer != "auto") {
    return(optimizer)
  }
  n <- nrow(x)
  p <- ncol(x)
  if (any(dropout > 0) || length(callbacks) > 0L) {
    return("adam")
  }
  layer_sizes <- c(p, hidden, output_dim)
  n_params <- sum(layer_sizes[-length(layer_sizes)] * layer_sizes[-1L]) +
    sum(layer_sizes[-1L])
  if (n <= 5000L && n_params <= 5000L) {
    return("lbfgs")
  }
  "adam"
}

.has_rcpp_backend <- function() {
  is.loaded("neuralnetwork_dense_bias", PACKAGE = "neuralnetwork")
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

.plot_network <- function(model, ...) {
  layer_sizes <- c(length(model$blueprint$feature_names), model$hidden,
                   model$output_dim)
  layer_x <- seq_along(layer_sizes)
  max_nodes <- max(layer_sizes)
  graphics::plot(NA, xlim = c(0.5, length(layer_sizes) + 0.5),
                 ylim = c(0.5, max_nodes + 0.5), axes = FALSE,
                 xlab = "", ylab = "", ...)
  graphics::box()
  coords <- vector("list", length(layer_sizes))
  for (i in seq_along(layer_sizes)) {
    y <- seq_len(layer_sizes[i])
    y <- y + (max_nodes - layer_sizes[i]) / 2
    coords[[i]] <- cbind(x = rep(layer_x[i], layer_sizes[i]), y = y)
    graphics::points(coords[[i]], pch = 21, bg = "white", cex = 1.5)
  }
  for (i in seq_len(length(layer_sizes) - 1L)) {
    left <- coords[[i]]
    right <- coords[[i + 1L]]
    for (a in seq_len(nrow(left))) {
      for (b in seq_len(nrow(right))) {
        graphics::segments(left[a, "x"], left[a, "y"],
                           right[b, "x"], right[b, "y"],
                           col = grDevices::adjustcolor("gray40", 0.25))
      }
    }
  }
  labels <- c("input", if (length(model$hidden)) paste0("hidden", seq_along(model$hidden)), "output")
  graphics::axis(1, at = layer_x, labels = labels, tick = FALSE)
  invisible(model)
}

.resolve_response_index <- function(model, response) {
  names <- if (model$task %in% c("classification", "binary_classification")) {
    model$classes
  } else {
    model$output_names
  }
  if (is.character(response)) {
    match <- match(response, names)
    if (is.na(match)) {
      stop("Unknown response: ", response, call. = FALSE)
    }
    return(match)
  }
  if (!is.numeric(response) || length(response) != 1L ||
      response < 1 || response > length(names)) {
    stop("`response` must name or index an output.", call. = FALSE)
  }
  as.integer(response)
}

.resolve_feature_index <- function(model, feature) {
  names <- model$blueprint$feature_names
  if (is.character(feature)) {
    match <- match(feature, names)
    if (is.na(match)) {
      stop("Unknown feature: ", feature, call. = FALSE)
    }
    return(match)
  }
  if (!is.numeric(feature) || length(feature) != 1L ||
      feature < 1 || feature > length(names)) {
    stop("Feature must name or index a predictor.", call. = FALSE)
  }
  as.integer(feature)
}

.predict_matrix_response <- function(model, x, response) {
  out <- .forward(x, model$params, model$activation, model$task, dropout = 0,
                  training = FALSE, backend = model$backend %||% "r")$output
  if (model$task == "binary_classification") {
    prob <- cbind(1 - out[, 1L], out[, 1L])
    return(prob[, response])
  }
  if (model$task == "classification") {
    return(out[, response])
  }
  out <- .inverse_target_scaler(out, model$target_scaler %||% NULL)
  out[, response]
}

.encode_truth_for_model <- function(model, truth) {
  if (model$task == "classification") {
    truth <- factor(truth, levels = model$classes)
    if (anyNA(truth)) {
      stop("Truth contains classes not seen during fitting.", call. = FALSE)
    }
    return(nn_class_ind(truth))
  }
  if (model$task == "binary_classification") {
    truth <- factor(truth, levels = model$classes)
    if (anyNA(truth)) {
      stop("Truth contains classes not seen during fitting.", call. = FALSE)
    }
    return(matrix(as.integer(truth) - 1L, ncol = 1L))
  }
  .apply_target_scaler(.double_matrix(truth), model$target_scaler %||% NULL)
}

.flatten_params <- function(params) {
  values <- numeric()
  labels <- character()
  for (layer in seq_along(params$W)) {
    w <- params$W[[layer]]
    b <- params$b[[layer]]
    w_rows <- row(w)
    w_cols <- col(w)
    values <- c(values, as.vector(w), as.vector(b))
    labels <- c(
      labels,
      paste0("W", layer, "[", as.vector(w_rows), ",", as.vector(w_cols), "]"),
      paste0("b", layer, "[", seq_along(b), "]")
    )
  }
  list(values = values, names = labels)
}

.flatten_grads <- function(grads) {
  values <- numeric()
  for (layer in seq_along(grads$W)) {
    values <- c(values, as.vector(grads$W[[layer]]), as.vector(grads$b[[layer]]))
  }
  values
}

.unflatten_params <- function(values, template) {
  params <- .copy_params(template)
  pos <- 1L
  for (layer in seq_along(params$W)) {
    w_len <- length(params$W[[layer]])
    b_len <- length(params$b[[layer]])
    params$W[[layer]][] <- values[pos:(pos + w_len - 1L)]
    pos <- pos + w_len
    params$b[[layer]][] <- values[pos:(pos + b_len - 1L)]
    pos <- pos + b_len
  }
  params
}

.fit_lbfgs <- function(params, x_train, y_train, y_train_raw,
                       x_val, y_val, y_val_raw, activation, task, l2,
                       loss, huber_delta, backend, target_scaler,
                       weight_train, weight_val, maxit, verbose) {
  template <- .copy_params(params)
  initial <- .flatten_params(params)

  fn <- function(theta) {
    current <- .unflatten_params(theta, template)
    .dataset_loss(x_train, y_train, current, activation, task, l2,
                  backend, weight_train, loss, huber_delta)
  }

  gr <- function(theta) {
    current <- .unflatten_params(theta, template)
    forward <- .forward(x_train, current, activation, task, dropout = 0,
                        training = FALSE, backend = backend)
    grads <- .backward(forward, current, y_train, activation, task, l2,
                       weights = weight_train, loss = loss,
                       huber_delta = huber_delta)
    .flatten_grads(grads)
  }

  result <- stats::optim(
    par = initial$values,
    fn = fn,
    gr = gr,
    method = "L-BFGS-B",
    control = list(maxit = maxit, trace = if (isTRUE(verbose)) 1 else 0)
  )

  params <- .unflatten_params(result$par, template)
  train_stats <- .dataset_stats(x_train, y_train, y_train_raw, params,
                                activation, task, l2, backend,
                                target_scaler, weight_train, loss,
                                huber_delta)
  validation_stats <- if (nrow(x_val) > 0L) {
    .dataset_stats(x_val, y_val, y_val_raw, params, activation, task, l2,
                   backend, target_scaler, weight_val, loss, huber_delta)
  } else {
    list(loss = NA_real_, metric = NA_real_)
  }
  final_grads <- gr(result$par)
  history <- data.frame(
    epoch = unname(result$counts[["function"]]),
    train_loss = train_stats$loss,
    validation_loss = validation_stats$loss,
    train_metric = train_stats$metric,
    validation_metric = validation_stats$metric,
    gradient_norm = sqrt(sum(final_grads^2)),
    learning_rate = NA_real_,
    backtracked = FALSE
  )
  list(
    params = params,
    history = history,
    best_loss = if (is.na(validation_stats$loss)) train_stats$loss else validation_stats$loss,
    convergence = result$convergence,
    message = result$message
  )
}

.should_report <- function(epoch, epochs) {
  epoch == 1L || epoch == epochs || epoch %% max(1L, floor(epochs / 10L)) == 0L
}

.report_epoch <- function(epoch, train_loss, validation_loss) {
  if (is.na(validation_loss)) {
    message(sprintf("epoch %d | train loss %.5f", epoch, train_loss))
  } else {
    message(sprintf("epoch %d | train loss %.5f | validation loss %.5f",
                    epoch, train_loss, validation_loss))
  }
}
