\name{neuralnetwork-metrics}
\alias{neuralnetwork-metrics}
\title{Metrics used by neuralnetwork}
\description{
Metric definitions for model evaluation and selection.

The same conventions apply in \code{\link{nn_evaluate}},
\code{\link{nn_tune}} and \code{\link{nn_cv}}.

\link[=nn_permutation_importance]{Permutation importance} uses them too.
}
\details{
Classification metrics:

\describe{
\item{\code{accuracy}}{Fraction of correct predictions.}
\item{\code{balanced_accuracy}}{Mean recall across classes present in the truth
data. Useful when class frequencies are uneven.}
\item{\code{macro_precision}}{Mean class-wise precision over all fitted classes.}
\item{\code{macro_recall}}{Mean class-wise recall over all fitted classes.}
\item{\code{macro_f1}}{Mean class-wise F1 score over all fitted classes.}
\item{\code{log_loss}}{Negative log likelihood of the true class under the
predicted probabilities. Lower is better.}
}

Binary classification also reports:

\describe{
\item{\code{sensitivity}}{Recall for the second fitted class, treated as the
positive class.}
\item{\code{specificity}}{Recall for the first fitted class.}
\item{\code{precision}}{Positive predictive value for the second fitted class.}
\item{\code{recall}}{Alias for sensitivity.}
\item{\code{f1}}{Positive-class F1, computed as
\eqn{2 TP / (2 TP + FP + FN)}.}
}

For a class with no predicted observations, precision is set to zero. For a
class absent from the truth, recall is set to zero. F1 is computed directly as
\eqn{2 TP / (2 TP + FP + FN)}, with zero returned when the denominator is zero.
These rules apply to binary metrics too. They are reporting conventions;
a class absent from the truth has no observed sensitivity to estimate.

Macro averages include every class learned during fitting, even when a class
is absent from the evaluation data or predictions. A class present in the
truth but never predicted therefore contributes zero to macro F1 rather than
being dropped. Balanced accuracy averages only over classes present in the
truth. It equals macro recall when every fitted class appears in the truth.
Macro F1 averages the individual class F1 scores; it is not the harmonic mean
of macro precision and macro recall.

Regression metrics:

\describe{
\item{\code{rmse}}{Root mean squared error. Lower is better.}
\item{\code{mae}}{Mean absolute error. Lower is better.}
\item{\code{rsq}}{Pooled coefficient of determination. Higher is better.
Returns \code{NA} for constant truth, including a single observation.}
}

For model selection helpers, \code{metric = "auto"} uses accuracy for
classification and RMSE for regression. \code{metric = "f1"} uses
positive-class F1 for binary classification and macro F1 for multiclass
classification. \code{metric = "loss"} uses the unpenalized fitted data
loss in every helper, including Huber loss when selected. Regression loss is
computed on the training target scale and sums over outputs; RMSE and MAE
are averages over both rows and outputs on the response scale. They are
different quantities, not aliases.

For \eqn{q} regression outputs and row weights \eqn{w_i}, RMSE is
\eqn{\sqrt{\sum_i w_i \sum_j (\hat y_{ij}-y_{ij})^2 / (q \sum_i w_i)}}.
MAE replaces squared errors with absolute errors. R-squared is one minus the
ratio of weighted total squared error to weighted total variation, centering
each outcome column at its own weighted mean. This pools output variances;
it does not equally average the individual outputs' R-squared values.
Different units or scales can make one output dominate pooled R-squared.

Sample-weighted classification uses weighted confusion counts. Classes with
zero true weight are excluded from balanced accuracy but not macro averages.
Reported log loss clips probabilities below \eqn{10^{-15}}; training and
\code{metric = "loss"} use stable logits without that cap. For soft targets,
cross-entropy sums target probability times negative log predicted probability.
}
\seealso{
\code{\link{nn_evaluate}}, \code{\link{nn_tune}}, \code{\link{nn_cv}},
\code{\link{nn_permutation_importance}}
}
\examples{
class_names <- c("Excellent", "Poor", "Typical")
confusion <- matrix(
  c(20, 0, 45, 0, 0, 19, 18, 0, 378),
  nrow = 3, byrow = TRUE,
  dimnames = list(truth = class_names, estimate = class_names)
)
confusion
class_counts <- rowSums(confusion) + colSums(confusion)
class_f1 <- 2 * diag(confusion) / class_counts
class_f1
mean(class_f1) # 0.4301658; Poor contributes zero, so divide by three
}
