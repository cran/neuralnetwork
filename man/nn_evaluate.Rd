\name{nn_evaluate}
\alias{nn_evaluate}
\title{Evaluate a neuralnetwork model}
\description{
Compute regression or classification metrics for a fitted model. Classification
metrics include accuracy, balanced accuracy, log loss, macro precision, macro
recall, and macro F1. Binary classification also reports sensitivity,
specificity, precision, recall, and F1.
}
\usage{
nn_evaluate(model, newdata, y = NULL)
}
\arguments{
\item{model}{A fitted \code{neuralnetwork} object.}
\item{newdata}{Data containing predictors, and optionally the response.}
\item{y}{Optional truth vector or matrix. Required when \code{newdata} does not
contain the original formula response.}
}
\details{
For classification, truth values are compared against predicted classes and
probabilities. Multiclass metrics are macro-averaged across classes. For binary
classification, sensitivity, specificity, precision, recall, and F1 use the
second fitted class as the positive class.

For regression, predictions are returned on the original response scale before
metrics are computed. Regression metrics are RMSE, MAE, and R-squared.
}
\value{
An object of class \code{neuralnetwork_evaluation}, a list with:
\describe{
\item{\code{metrics}}{Named numeric vector of task-aware metrics.}
\item{\code{predictions}}{Predicted classes for classification, or numeric
predictions for regression.}
\item{\code{confusion}}{Classification only: a confusion table.}
\item{\code{probabilities}}{Classification only: class probabilities.}
}
}
\seealso{
\code{\link{neuralnetwork-metrics}}, \code{\link{nn_tune}},
\code{\link{nn_cv}}, \code{\link{nn_permutation_importance}}
}
\examples{
fit <- nn_fit(Species ~ ., iris, hidden = 6, epochs = 5,
              verbose = FALSE, seed = 1)
nn_evaluate(fit, iris)
}
