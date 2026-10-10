# neuralnetwork 0.1.4 (unreleased)

This update includes the unreleased 0.1.2 and 0.1.3 corrections. Users upgrading
from 0.1.1 should recompute classification metrics and refit models affected by
the weighted training, formula preprocessing or checkpoint-selection fixes.

## Training and numerical stability

- Fixed weighted mini-batch gradients: weights no longer cancel in one-row
  batches. The normalization now matches the full weighted objective.
- Excluded zero-weight training rows from optimization, scalers and fitted
  formula bases. Fitted metadata records how many training rows were excluded.
- Finite gradient clipping now selects Adam automatically; explicit L-BFGS
  rejects the unsupported control instead of silently ignoring it.
- Shared L-BFGS loss/gradient forward passes and allocated optimizer state
  only when needed. These changes do not add runtime dependencies.
- Avoided cancellation in near-hard binary soft-target loss, and intermediate
  overflow/underflow in extreme multiclass loss and L2 gradients.
- Preserved RPROP direction-change detection for extremely small gradients by
  comparing signs rather than multiplying gradient magnitudes.
- Clarified errors for complex input and hidden-layer integers outside the
  supported range; rejected non-finite scaling before native kernels.
- Added deep-network gradient and weighted batch-average oracles, one-row
  weight checks, zero-weight invariance, cache instrumentation and allocation
  tests. Prior numerical and public-interface tests remain in the suite.
- Wrapped reference examples and help-page text for wider code fonts as well
  as the macOS manual font. No API defaults changed.

## Evaluation, formulas and model selection

- Corrected multi-output R-squared centering and aligned training RMSE with
  evaluation RMSE. Constant truth returns an undefined R-squared explicitly.
- Preserved fitted formula bases and contrasts, applied response transformations
  during evaluation, and learned holdout bases from training rows only.
- Rejected mismatched response rows and missing named predictor columns; aligned
  named outcome matrices and accepted classification indicator/soft-target truth.
- Added sample-weighted evaluation and importance. Cross-validation now splits
  weights, stratifies encoded classes, retains undefined folds and reports score
  counts. All helpers use the fitted unpenalized data loss for `metric = "loss"`.
- Tuning now compares candidates on one common holdout and ranks only finite
  scores. Candidate identifiers preserve the link to stored models.
- Used stable logits for classification objectives so saturated losses agree
  with gradients. Fixed the mini-batch L2 denominator, skipped zero-weight
  batches, rejected non-finite optimizer states and separated checkpoint saving
  from the patience improvement threshold.
- Removed fabricated intervals from singular Hessians. Parameter intervals are
  restricted to identifiable unweighted, unregularized no-hidden-layer linear
  or binary logistic L-BFGS fits, using total information and proper dispersion.
- Corrected sensitivities to use pre-scaling input units, fixed intercept-only
  parameter serialization and single-row permutations, and validated saved model
  structure on load. Unsupported L-BFGS epoch controls now fail explicitly.
- Added independent gradient, optimizer, regression, formula, selection and
  inference tests; tightened the release checker to reject any unapproved NOTE.
- Corrected plotting labels, included both loss series in the vertical range,
  drew single stored diagnostics as points, and stopped sensitivity plots from
  magnifying numerical noise in nearly constant derivatives.

## Missed-class metrics

- Fixed inflated macro F1 and macro precision when a fitted class has no
  predicted observations. F1 now uses confusion-matrix counts directly, so
  wholly missed classes also contribute zero. Thanks to Graham Upton for the
  reproducible report.
- Macro precision, recall, and F1 now average over all fitted classes, using
  zero for a class-wise zero denominator. Binary metrics use the same rule;
  balanced accuracy still averages recall over classes present in the truth.
- Added regression tests for the reported matrix, 591 exhaustive binary and
  multiclass tables, and both prediction backends and model-selection paths.
- Made binary model-selection scores agree with prediction at a probability
  of exactly 0.5, and reject empty classification evaluations with a clear error.
- Documented the class averaging and zero-denominator rules with a worked
  confusion-matrix example.
- Clarified that printed fit metrics describe the selected checkpoint, not
  necessarily the last epoch.

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
