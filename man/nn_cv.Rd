\name{nn_cv}
\alias{nn_cv}
\title{Cross-validate neuralnetwork models}
\description{
Run repeated k-fold cross-validation for \code{\link{nn_fit}} models and return
fold-level scores plus summary statistics.
}
\usage{
nn_cv(
  x,
  y = NULL,
  data = NULL,
  k = 5,
  repeats = 1,
  stratify = TRUE,
  metric = c("auto", "loss", "accuracy", "balanced_accuracy",
             "log_loss", "f1", "rmse", "mae", "rsq"),
  seed = NULL,
  verbose = FALSE,
  ...
)
}
\arguments{
\item{x, y, data}{Model inputs passed to \code{\link{nn_fit}}.}
\item{k}{Integer number of folds, between 2 and the number of rows.}
\item{repeats}{Positive integer number of repeated fold assignments.}
\item{stratify}{Whether to stratify classification folds, including indicator
matrices when \code{task = "classification"}. Soft targets use their first
maximal class for stratification.}
\item{metric}{Metric to report. \code{"auto"} uses accuracy for classification
and RMSE for regression. \code{"f1"} uses the positive-class F1 for binary
classification and macro F1 for multiclass classification.}
\item{seed}{Optional non-negative integer random seed. Fold models receive
deterministic consecutive seeds derived from this value.}
\item{verbose}{Whether individual fits should print progress.}
\item{...}{Additional arguments passed to \code{\link{nn_fit}}.}
}
\details{
For classification outcomes, \code{stratify = TRUE} attempts to
keep class proportions similar across folds. Each class must have at least
\code{k} rows; otherwise a stratified split would create folds whose training
sets do not contain all classes. Unused outcome levels do not count as
observed classes. For regression outcomes, folds are sampled without
stratification. Repeated cross-validation creates a new fold assignment for
each repeat.

The metric is computed on each held-out fold using \code{\link{nn_evaluate}}.
The returned \code{summary} data frame reports the mean and standard deviation
of finite fold scores for each metric name, with \code{n_scored} and
\code{n_folds}. Undefined scores remain in \code{results}; an entirely
undefined metric returns \code{NA} mean and standard deviation, not an empty
summary. Fold-score standard deviation is not a confidence interval.
\code{sample_weight} in \code{...} is split into training and held-out
weights; held-out metrics are weighted too. \code{metric = "loss"} uses the
fitted data loss without a regularization penalty, not RMSE.
}
\value{
A \code{neuralnetwork_cv} object, a list with:
\describe{
\item{\code{results}}{Fold-level scores.}
\item{\code{summary}}{Mean and standard deviation by metric.}
\item{\code{models}}{Fitted model for each fold.}
\item{\code{folds}}{Repeat identifier, fold identifier and held-out row indices
for each model.}
\item{\code{k}}{Number of folds.}
\item{\code{repeats}}{Number of repeats.}
\item{\code{call}}{Matched call.}
}
}
\seealso{
\code{\link{neuralnetwork-metrics}}, \code{\link{nn_fit}},
\code{\link{nn_tune}}
}
\examples{
cv <- nn_cv(Species ~ ., iris, k = 3, epochs = 3,
            seed = 1, verbose = FALSE)
cv
}
