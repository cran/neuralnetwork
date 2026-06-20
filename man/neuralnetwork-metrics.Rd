\name{neuralnetwork-metrics}
\alias{neuralnetwork-metrics}
\title{Metrics used by neuralnetwork}
\description{
Reference for the metrics returned by \code{\link{nn_evaluate}} and accepted by
\code{\link{nn_tune}}, \code{\link{nn_cv}}, and
\code{\link{nn_permutation_importance}}.
}
\details{
Classification metrics:

\describe{
\item{\code{accuracy}}{Fraction of correct predictions.}
\item{\code{balanced_accuracy}}{Mean recall across classes. Useful when class
frequencies are uneven.}
\item{\code{macro_precision}}{Mean class-wise precision.}
\item{\code{macro_recall}}{Mean class-wise recall.}
\item{\code{macro_f1}}{Mean class-wise F1 score.}
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
\item{\code{f1}}{Positive-class harmonic mean of precision and recall.}
}

Regression metrics:

\describe{
\item{\code{rmse}}{Root mean squared error. Lower is better.}
\item{\code{mae}}{Mean absolute error. Lower is better.}
\item{\code{rsq}}{Coefficient of determination. Higher is better.}
}

For model selection helpers, \code{metric = "auto"} uses accuracy for
classification and RMSE for regression. \code{metric = "f1"} uses
positive-class F1 for binary classification and macro F1 for multiclass
classification. \code{metric = "loss"} uses log loss for classification and
the fitted loss where the helper evaluates the model's encoded output.
}
\seealso{
\code{\link{nn_evaluate}}, \code{\link{nn_tune}}, \code{\link{nn_cv}},
\code{\link{nn_permutation_importance}}
}
