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
- Added compatibility helpers for common `nnet` and `neuralnet` workflows.
- Added a practical workflow vignette.
