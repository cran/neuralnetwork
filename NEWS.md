# neuralnetwork 0.1.1

Maintenance release.

- Added optimizer convergence metadata to fitted models. L-BFGS fits now expose
  the convergence code and message returned by `stats::optim()`, and
  `summary()` prints that diagnostic.
- Added an explicit stratified cross-validation guard: when `stratify = TRUE`,
  each class must have at least `k` rows. This fails early with a clear message
  instead of producing a later fold-specific class mismatch.
- Added `error_action = "continue"` to `nn_tune()` so exploratory grids can
  record failed candidates and continue ranking the usable fits.
- Added explicit outcome-length checks for evaluation, permutation importance,
  and finite-difference Hessian helpers.
- Improved prediction errors for missing predictors and factor levels not seen
  during fitting.
- Added validation for classification target matrices and for missing or empty
  inputs to `nn_class_ind()`.
- Added explicit validation for finite-difference step sizes, Hessian parameter
  limits, and confidence levels.
- Added stricter validation for optimizer hyperparameters, random seeds,
  integer loop counts, and callback return values.
- Clarified L-BFGS printed output so it reports function evaluations rather
  than epochs.
- Hardened the local CRAN check script so it parses `R CMD check` status lines
  and treats ERROR/WARNING results as failures.
- Added tests and documentation for the new diagnostics and cross-validation
  validation rule.

# neuralnetwork 0.1.0

First CRAN-oriented release.

- Added `nn_fit()` for compact multilayer perceptrons with formula, data frame,
  matrix, and vector inputs.
- Added regression, binary classification, and multiclass classification.
- Added Adam, SGD, momentum, Nesterov, RPROP, GRPROP, and L-BFGS optimizers.
- Added automatic hidden-layer sizing, optimizer selection, and activation
  selection.
- Added optional portable Rcpp forward-pass kernels.
- Added dropout, L2 regularization, gradient clipping, learning-rate decay,
  validation splits, early stopping, and callback hooks.
- Added sample weights and balanced class weights.
- Added robust Huber loss for regression.
- Added task-aware evaluation metrics, tuning, repeated k-fold
  cross-validation, and permutation importance.
- Added save/load helpers and S3 methods for prediction, printing, plotting,
  summaries, and coefficients.
- Added compatibility helpers for common `nnet` and `neuralnet` tasks.
- Added a worked-example vignette.
