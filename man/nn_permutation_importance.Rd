\name{nn_permutation_importance}
\alias{nn_permutation_importance}
\title{Permutation feature importance}
\description{
Estimate feature importance by repeatedly permuting each encoded model input
and measuring the change in model performance. Positive importance means the
permuted feature made the selected metric worse.
}
\usage{
nn_permutation_importance(
  model,
  newdata,
  y = NULL,
  metric = c("auto", "loss", "accuracy", "balanced_accuracy",
             "log_loss", "f1", "rmse", "mae", "rsq"),
  n_repeats = 5,
  seed = NULL
)
}
\arguments{
\item{model}{A fitted \code{neuralnetwork} object.}
\item{newdata}{Data containing predictors, and optionally the response.}
\item{y}{Optional truth vector or matrix. Required when \code{newdata} does not
contain the original formula response.}
\item{metric}{Metric used to score the permutations.}
\item{n_repeats}{Number of permutations per feature.}
\item{seed}{Optional random seed.}
}
\details{
The baseline metric is computed once on \code{newdata}. Then each encoded model
input column is shuffled \code{n_repeats} times and scored again. Positive
importance means the permutation worsened the metric. For error-like metrics
such as loss, RMSE, MAE, and log loss, larger permuted values are worse. For
score-like metrics such as accuracy, balanced accuracy, F1, macro F1, and
R-squared, smaller permuted values are worse.

When a formula or data-frame model contains factor predictors, importance is
computed on the encoded model-matrix columns rather than on the original factor
as a single group.
}
\value{
A data frame of class \code{neuralnetwork_importance}, sorted by decreasing
importance, with columns:
\describe{
\item{\code{feature}}{Encoded feature name.}
\item{\code{importance}}{Metric degradation caused by permutation.}
\item{\code{baseline}}{Baseline metric before permutation.}
\item{\code{permuted}}{Mean metric after permutation.}
\item{\code{metric}}{Metric used for scoring.}
\item{\code{n_repeats}}{Number of permutations per feature.}
}
}
\seealso{
\code{\link{neuralnetwork-metrics}}, \code{\link{nn_evaluate}}
}
\examples{
fit <- nn_fit(mpg ~ wt + hp, mtcars, hidden = 4, epochs = 5,
              verbose = FALSE, seed = 1)
nn_permutation_importance(fit, mtcars, n_repeats = 2, seed = 2)
}
