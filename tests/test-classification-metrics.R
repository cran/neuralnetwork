library(neuralnetwork)

indices_from_confusion <- function(confusion) {
  cells <- which(confusion > 0, arr.ind = TRUE)
  list(
    truth = rep(cells[, 1L], times = confusion[cells]),
    estimate = rep(cells[, 2L], times = confusion[cells])
  )
}

probabilities_for <- function(estimate, k) {
  prob <- matrix(0.2 / (k - 1L), nrow = length(estimate), ncol = k)
  prob[cbind(seq_along(estimate), estimate)] <- 0.8
  prob
}

# Use one-vs-rest subsets as an independent reference for the count-based code.
reference_metrics <- function(truth, estimate, prob, k) {
  precision <- recall <- f1 <- numeric(k)
  for (i in seq_len(k)) {
    predicted <- estimate == i
    observed <- truth == i
    precision[i] <- if (any(predicted)) mean(truth[predicted] == i) else 0
    recall[i] <- if (any(observed)) mean(estimate[observed] == i) else 0
    f1[i] <- if (precision[i] == 0 || recall[i] == 0) {
      0
    } else {
      2 / (1 / precision[i] + 1 / recall[i])
    }
  }
  metrics <- c(
    accuracy = mean(truth == estimate),
    balanced_accuracy = mean(recall[seq_len(k) %in% truth]),
    macro_precision = mean(precision),
    macro_recall = mean(recall),
    macro_f1 = mean(f1),
    log_loss = -mean(log(pmax(prob[cbind(seq_along(truth), truth)], 1e-15)))
  )
  if (k == 2L) {
    metrics <- c(metrics, sensitivity = recall[2L], specificity = recall[1L],
                 precision = precision[2L], recall = recall[2L], f1 = f1[2L])
  }
  metrics
}

check_metrics <- function(actual, expected) {
  stopifnot(identical(names(actual), names(expected)))
  stopifnot(all(is.finite(actual)))
  stopifnot(isTRUE(all.equal(actual, expected, tolerance = 1e-12)))
}

reported_confusion <- matrix(c(20, 0, 45, 0, 0, 19, 18, 0, 378),
                             nrow = 3, byrow = TRUE)
reported_classes <- c("Excellent", "Poor", "Typical")
reported <- indices_from_confusion(reported_confusion)
reported_prob <- probabilities_for(reported$estimate, 3L)
reported_metrics <- neuralnetwork:::.classification_metrics_from_indices(
  reported$truth, reported$estimate, reported_prob, reported_classes
)
stopifnot(abs(reported_metrics[["macro_f1"]] - (40 / 103 + 756 / 838) / 3) < 1e-12)
stopifnot(abs(reported_metrics[["macro_precision"]] -
                (20 / 38 + 378 / 442) / 3) < 1e-12)
check_metrics(reported_metrics, reference_metrics(
  reported$truth, reported$estimate, reported_prob, 3L
))

# Exhaust all nonempty small tables, including absent and wholly missed classes.
table_count <- 0L
for (k in 2:3) {
  cells <- if (k == 2L) 0:2 else 0:1
  tables <- expand.grid(rep(list(cells), k * k))
  classes <- paste0("class", seq_len(k))
  for (row in seq_len(nrow(tables))) {
    confusion <- matrix(as.numeric(tables[row, ]), nrow = k)
    if (sum(confusion) == 0) next
    indices <- indices_from_confusion(confusion)
    prob <- probabilities_for(indices$estimate, k)
    actual <- neuralnetwork:::.classification_metrics_from_indices(
      indices$truth, indices$estimate, prob, classes
    )
    check_metrics(actual, reference_metrics(indices$truth, indices$estimate,
                                           prob, k))
    table_count <- table_count + 1L
  }
}
stopifnot(table_count == 591L)

# Integer weights are an independent observation-replication oracle.
set.seed(827)
for (k in 2:6) for (rep in 1:10) {
  truth <- sample.int(k, 20, replace = TRUE)
  estimate <- sample.int(k, 20, replace = TRUE)
  weights <- sample(0:4, 20, replace = TRUE)
  weights[1L] <- 1L
  prob <- probabilities_for(estimate, k)
  rows <- rep(seq_len(20), times = weights)
  weighted <- neuralnetwork:::.classification_metrics_from_indices(
    truth, estimate, prob, paste0("class", seq_len(k)), weights)
  check_metrics(weighted, reference_metrics(truth[rows], estimate[rows], prob[rows, ], k))
}

# Exercise the public evaluation API and the shared model-selection paths.
large <- neuralnetwork:::.classification_metrics_from_indices(
  c(2L, 1L), c(2L, 1L), matrix(c(0.2, 0.8, 0.8, 0.2), 2), c("no", "yes"), c(1e308, 0))
stopifnot(abs(large["f1"] - 1) < 1e-12, abs(large["log_loss"] + log(0.8)) < 1e-12)
large <- neuralnetwork:::.classification_metrics_from_indices(
  c(2L, 1L), c(1L, 1L), matrix(c(0.8, 0.8, 0.2, 0.2), 2), c("no", "yes"), c(1e308, 0))
stopifnot(abs(large["log_loss"] + log(0.2)) < 1e-12)
mixed <- neuralnetwork:::.classification_metrics_from_indices(
  c(2L, 1L), c(2L, 1L), matrix(c(0.2, 0.8, 0.8, 0.2), 2), c("no", "yes"), c(1e308, 1e-308))
stopifnot(abs(mixed["balanced_accuracy"] - 1) < 1e-12, abs(mixed["macro_f1"] - 1) < 1e-12)
for (k in 2:3) {
  classes <- if (k == 3L) reported_classes else c("Negative", "Positive")
  model <- nn_fit(diag(k), factor(classes, levels = classes), hidden = 0,
                  optimizer = "sgd", scale = FALSE, epochs = 1, seed = 13,
                  backend = "r", verbose = FALSE)
  model$params$W[[1L]] <- if (k == 2L) matrix(c(-2, 2), ncol = 1) else 2 * diag(k)
  model$params$b[[1L]][] <- 0
  cases <- if (k == 2L) {
    list(matrix(c(0, 2, 3, 0), 2, byrow = TRUE),
         matrix(c(3, 0, 2, 0), 2, byrow = TRUE),
         matrix(c(3, 0, 0, 0), 2, byrow = TRUE),
         matrix(c(0, 0, 0, 3), 2, byrow = TRUE))
  } else {
    list(reported_confusion, diag(c(2, 0, 3)),
         matrix(c(0, 2, 0, 0, 0, 2, 2, 0, 0), 3, byrow = TRUE))
  }
  for (backend in c("r", "rcpp")) {
    model$backend <- backend
    for (confusion in cases) {
      indices <- indices_from_confusion(confusion)
      x <- diag(k)[indices$estimate, , drop = FALSE]
      truth <- factor(classes[indices$truth], levels = classes)
      evaluation <- nn_evaluate(model, x, y = truth)
      check_metrics(evaluation$metrics, reference_metrics(
        indices$truth, indices$estimate, evaluation$probabilities, k
      ))
      stopifnot(all(unname(evaluation$confusion) == confusion))
      target <- if (k == 2L) {
        matrix(as.numeric(indices$truth == 2L), ncol = 1)
      } else {
        diag(k)[indices$truth, , drop = FALSE]
      }
      output <- if (k == 2L) {
        evaluation$probabilities[, 2L, drop = FALSE]
      } else {
        evaluation$probabilities
      }
      expected_f1 <- evaluation$metrics[[if (k == 2L) "f1" else "macro_f1"]]
      check_metrics(neuralnetwork:::.classification_metrics(
        truth, evaluation$predictions, evaluation$probabilities, classes
      ), evaluation$metrics)
      stopifnot(abs(neuralnetwork:::.score_output_metric(
        output, target, model, "f1"
      )$value - expected_f1) < 1e-12)
      stopifnot(abs(neuralnetwork:::.score_model_matrix(
        model, x, truth, "f1"
      )$value - expected_f1) < 1e-12)
    }
    if (k == 2L) {
      # A probability of exactly 0.5 must select the positive class everywhere.
      model$params$W[[1L]][] <- 0
      x <- diag(k)
      truth <- factor(classes, levels = classes)
      evaluation <- nn_evaluate(model, x, y = truth)
      stopifnot(all(evaluation$predictions == "Positive"))
      stopifnot(abs(evaluation$metrics[["f1"]] - 2 / 3) < 1e-12)
      for (metric in c("accuracy", "balanced_accuracy", "f1")) {
        expected <- evaluation$metrics[[metric]]
        stopifnot(abs(neuralnetwork:::.score_output_metric(
          matrix(0.5, 2, 1), matrix(c(0, 1), 2, 1), model, metric
        )$value - expected) < 1e-12)
        stopifnot(abs(neuralnetwork:::.score_model_matrix(
          model, x, truth, metric
        )$value - expected) < 1e-12)
      }
      empty_error <- tryCatch({
        nn_evaluate(model, matrix(numeric(), 0, k), y = character())
        NULL
      }, error = function(e) conditionMessage(e))
      stopifnot(identical(empty_error,
                          "Evaluation requires at least one observation."))
      model$params$W[[1L]] <- matrix(c(-2, 2), ncol = 1)
    }
  }
}

cat("Classification metrics: reported matrix, 591 exhaustive tables, and",
    "14 public evaluation/model-selection cases passed;",
    "binary threshold ties and empty evaluations checked.\n")
