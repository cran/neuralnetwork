# neuralnetwork

`neuralnetwork` fits multilayer perceptrons for tabular data in R. It accepts
formulas, data frames, matrices, and vectors; handles regression and
classification; and includes tuning, cross-validation, metrics, feature
importance, and model save/load helpers. It is meant for users who need more
than `nnet`'s single-hidden-layer interface or `neuralnet`'s manual training
style, but do not want to bring in a full deep-learning stack.

## Install

```r
install.packages("neuralnetwork")
```

To install the local source tarball:

```r
install.packages("neuralnetwork_0.1.4.tar.gz", repos = NULL, type = "source")
```

## Quick start

```r
library(neuralnetwork)

fit <- nn_fit(
  Species ~ .,
  data = iris,
  hidden = "auto",
  optimizer = "auto",
  epochs = 20,
  validation_split = 0.2,
  seed = 1,
  verbose = FALSE
)

fit
predict(fit, iris[1:5, ], type = "class")
round(predict(fit, iris[1:5, ], type = "prob"), 3)

ev <- nn_evaluate(fit, iris)
ev
```

The printed model reports the architecture, optimizer, loss, backend, training
length, and the selected checkpoint's training and validation scores. These
scores need not come from the last epoch. Epoch optimizers store their trajectory
in `fit$history`; L-BFGS stores final diagnostics in one row.
`nn_evaluate()` returns the metrics as a named vector and prints a compact
confusion matrix for classification.

This call scores all of `iris`, including the rows used for training. It checks
the workflow, not performance on new observations. Use `nn_cv()` for fold-level
estimates, and reserve an untouched test set before tuning for a final score.

Macro precision, recall, and F1 include all fitted classes. A class that is
never predicted contributes zero instead of disappearing from the average.
Class-wise zero denominators return zero. Balanced accuracy averages recall
over classes present in the truth, so it can differ from macro recall when an
evaluation set omits a fitted class. See `help("neuralnetwork-metrics")` for
the formulas and a worked example.

Row and class weights remain effective with one-row mini-batches. The data
gradient uses the full training-weight sum rather than normalizing each batch
by its own weights. Zero-weight training rows are omitted from training,
scalers and fitted formula bases. Inputs must still pass encoding checks.
Finite gradient clipping selects Adam under `optimizer = "auto"`; explicit
L-BFGS rejects clipping rather than ignoring it.

## Regression

For regression, put a numeric response on the left side of the formula.
Training can scale the target internally; predictions are returned on the
original response scale.

```r
fit_reg <- nn_fit(
  mpg ~ wt + hp + disp,
  data = mtcars,
  hidden = c(8, 4),
  optimizer = "adam",
  epochs = 40,
  batch_size = 8,
  learning_rate = 0.01,
  validation_split = 0.2,
  seed = 2,
  verbose = FALSE
)

predict(fit_reg, mtcars[1:5, ])
nn_evaluate(fit_reg, mtcars)
```

For regression problems with outliers, use Huber loss:

```r
fit_huber <- nn_fit(
  mpg ~ wt + hp + disp,
  data = mtcars,
  hidden = c(8, 4),
  optimizer = "adam",
  loss = "huber",
  huber_delta = 1,
  epochs = 40,
  batch_size = 8,
  learning_rate = 0.01,
  seed = 3,
  verbose = FALSE
)
```

## Choosing settings

Reasonable first choices:

- `hidden = "auto"` for a small architecture chosen from the task and input
  width.
- `optimizer = "auto"` for L-BFGS on small deterministic problems and Adam
  when using stochastic features such as dropout or callbacks.
- For L-BFGS fits, `epochs` is the optimizer iteration limit and printed
  training length is reported as function evaluations.
- `validation_split = 0.2` for validation loss, early stopping, or
  validation-based tuning.
- `metric = "balanced_accuracy"` for imbalanced classification, `metric = "f1"`
  when the positive class is the focus, and `metric = "mae"` or
  `metric = "rmse"` for regression.
- `loss = "huber"` for regression data where outliers may dominate squared
  error.

## Tuning and validation

```r
tuned <- nn_tune(
  Species ~ .,
  data = iris,
  grid = list(
    hidden = list(4, c(6, 3)),
    learning_rate = c(0.01, 0.003)
  ),
  metric = "balanced_accuracy",
  epochs = 8,
  validation_split = 0.2,
  seed = 4,
  verbose = FALSE
)

tuned
tuned$best_model
```

For exploratory grids, `error_action = "continue"` keeps candidate failures in
the results table while ranking the usable fits.

```r
cv <- nn_cv(
  Species ~ .,
  data = iris,
  k = 3,
  metric = "f1",
  hidden = 4,
  epochs = 5,
  seed = 5,
  verbose = FALSE
)

cv
```

## Feature importance

```r
imp <- nn_permutation_importance(
  fit_reg,
  mtcars,
  metric = "mae",
  n_repeats = 3,
  seed = 6
)

imp
```

## Function map

| Need | Use |
|---|---|
| Fit a model | `nn_fit()` |
| Predict classes, probabilities, or numeric responses | `predict()` |
| Evaluate metrics | `nn_evaluate()` |
| Tune a grid | `nn_tune()` |
| Cross-validate | `nn_cv()` |
| Estimate feature importance | `nn_permutation_importance()` |
| Save and load | `nn_save()`, `nn_load()` |
| Use `nnet` / `neuralnet` style helpers | `nn_multinom()`, `nn_compute()`, `nn_generalized_weights()` |

## What's included

- Formula, data frame, matrix, and vector inputs.
- Regression, binary classification, and multiclass classification.
- Multiple hidden layers, including `hidden = 0` for no hidden layer.
- Adam, SGD, momentum, Nesterov, RPROP, GRPROP, and L-BFGS optimizers.
- Automatic hidden-layer sizing, activation choice, and optimizer choice.
- Portable Rcpp forward-pass kernels when available.
- Dropout, L2 regularization, gradient clipping, learning-rate decay,
  validation splits, early stopping, and callbacks.
- Sample weights and balanced class weights.
- Huber loss for robust regression.
- Accuracy, balanced accuracy, macro F1, log loss, RMSE, MAE, and R-squared.
- Tuning, repeated k-fold cross-validation, permutation importance, save/load
  helpers, and S3 `predict()`, `print()`, `plot()`, `summary()`, and `coef()`.
- Training history plus optimizer convergence metadata for inspecting fitted
  models, including L-BFGS convergence codes from `stats::optim()`.
- Compatibility helpers for common `nnet` and `neuralnet` tasks:
  `nn_multinom()`, `nn_class_ind()`, `nn_which_is_max()`, `nn_compute()`,
  `nn_generalized_weights()`, `nn_gwplot()`, `nn_hessian()`, and
  `nn_confint()`.

## Numerical conventions and limits

Evaluation, tuning and cross-validation use the same metric definitions.
Tuning candidates share one holdout, and cross-validation splits row weights
along with observations. Undefined scores remain visible rather than silently
disappearing. Named predictor and outcome columns must match the fitted model.
Formula transformations and contrasts are reused at prediction time.

Multi-output RMSE averages over rows and outputs. R-squared pools output
variation after centering each column separately; constant truth returns `NA`.
`metric = "loss"` uses unpenalized fitted data loss, including Huber loss,
on the training target scale. It is not an alias for RMSE.

`nn_confint()` is limited to converged, unweighted, unregularized no-hidden-layer
linear or binary logistic L-BFGS fits. It requires the original training data
and refuses singular or saturated fits. `nn_hessian()` is a curvature
diagnostic, not an automatic covariance estimator for a hidden-layer network.
Input sensitivities describe encoded-feature derivatives, not `neuralnet`'s
log-odds generalized weights. These helpers are not drop-in inferential
replacements for every function in `nnet` or `neuralnet`.

Run `vignette("neuralnetwork")` for the longer worked example.

Reference help inside R: `?neuralnetwork`, `?neuralnetwork-metrics`,
`?neuralnetwork-callbacks`, and `?neuralnetwork-objects`.
