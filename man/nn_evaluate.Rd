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
nn_evaluate(model, newdata, y = NULL, sample_weight = NULL)
}
\arguments{
\item{model}{A fitted \code{neuralnetwork} object.}
\item{newdata}{Data containing predictors, and optionally the response.
Evaluation requires at least one observation.}
\item{y}{Optional truth vector or matrix, with one row or value per prediction
row. Required when \code{newdata} does not contain the original formula
response.}
\item{sample_weight}{Optional finite non-negative row weights with positive
total weight. These affect metrics, confusion counts and data loss.}
}
\details{
For classification, truth values are compared against predicted classes and
probabilities. Macro precision, recall, and F1 average over all fitted classes.
Class-wise precision, recall, and F1 use zero for a zero denominator; a class
that is never predicted still contributes to the macro averages. F1 is
computed directly from true-positive, false-positive, and false-negative
counts. Balanced accuracy averages recall over classes present in the truth.
See \code{\link{neuralnetwork-metrics}} for formulas and a worked example.

For binary classification, the second fitted class is the positive class.
Sensitivity and recall refer to that class; specificity refers to the first
fitted class. Precision and F1 refer to the positive class.
The positive class is predicted when its probability is at least 0.5,
including exactly 0.5; model-selection helpers use the same threshold.

For regression, predictions are returned on the original response scale before
metrics are computed. Regression metrics are RMSE, MAE, and R-squared.
For a transformed formula response such as \code{log(y)}, that response
transformation is applied to truth as well: scores are on the log scale, not
the raw \code{y} scale. Named outcome matrices are aligned to fitted outputs.
Classification indicator or probability matrices are also accepted. Class
metrics use their first-maximal class; log loss retains soft probabilities.
}
\value{
An object of class \code{neuralnetwork_evaluation}, a list with:
\describe{
\item{\code{metrics}}{Named numeric vector of task-aware metrics.}
\item{\code{loss}}{Unpenalized fitted data loss, using the training target
scaler. Classification uses stable logits; unlike reported \code{log_loss},
this value is not capped by probability clipping.}
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
