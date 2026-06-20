# neuralnetwork

`neuralnetwork` fits compact multilayer perceptrons for everyday R workflows:
formula input, tabular data, regression, classification, tuning, cross
validation, and readable model objects. It is designed for problems where
`nnet` feels too small, `neuralnet` feels too manual, and a full deep-learning
framework would be more machinery than the job needs.

## Install

```r
install.packages("neuralnetwork")
```

To install the local source tarball:

```r
install.packages("neuralnetwork_0.1.0.tar.gz", repos = NULL, type = "source")
```

## Quick Start

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

The printed model gives the architecture, optimizer, loss, backend, training
length, final training score, and validation score when available. The
evaluation object prints metrics as a small table and includes the confusion
matrix for classification.

## Regression

Regression uses the same interface. Targets can be scaled during training and
returned on the original scale for prediction.

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

## Choosing Settings

Good first defaults:

- `hidden = "auto"` for a compact task-aware architecture.
- `optimizer = "auto"` for L-BFGS on small deterministic problems and Adam
  when using stochastic features such as dropout or callbacks.
- `validation_split = 0.2` when you want validation loss, early stopping, or
  validation-based tuning.
- `metric = "balanced_accuracy"` for imbalanced classification, `metric = "f1"`
  when the positive class matters, and `metric = "mae"` or `metric = "rmse"` for
  regression.
- `loss = "huber"` for regression data where outliers may dominate squared
  error.

## Tuning and Validation

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

## Feature Importance

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

## Function Map

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

## What Is Included

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
- Compatibility helpers for common `nnet` and `neuralnet` workflows:
  `nn_multinom()`, `nn_class_ind()`, `nn_which_is_max()`, `nn_compute()`,
  `nn_generalized_weights()`, `nn_gwplot()`, `nn_hessian()`, and
  `nn_confint()`.

See `vignette("neuralnetwork")` for a fuller walkthrough.

For reference-style help inside R, see `?neuralnetwork`,
`?neuralnetwork-metrics`, `?neuralnetwork-callbacks`, and
`?neuralnetwork-objects`.
