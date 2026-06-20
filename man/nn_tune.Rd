\name{nn_tune}
\alias{nn_tune}
\title{Tune neuralnetwork hyperparameters}
\description{
Run a compact grid search over \code{nn_fit()} arguments and rank candidates by
validation performance.
}
\usage{
nn_tune(
  x,
  y = NULL,
  data = NULL,
  grid = NULL,
  validation_split = 0.2,
  metric = c("auto", "loss", "accuracy", "balanced_accuracy",
             "log_loss", "f1", "rmse", "mae", "rsq"),
  maximize = NULL,
  seed = NULL,
  verbose = FALSE,
  ...
)
}
\arguments{
\item{x, y, data}{Model inputs passed to \code{\link{nn_fit}}.}
\item{grid}{Named list of candidate values. Values may be vectors or lists.}
\item{validation_split}{Validation fraction passed to \code{\link{nn_fit}}.}
\item{metric}{Metric used to choose the best model. \code{"f1"} uses the
positive-class F1 for binary classification and macro F1 for multiclass
classification.}
\item{maximize}{Whether larger metric values are better. Inferred by default.}
\item{seed}{Optional seed.}
\item{verbose}{Whether individual fits should print progress.}
\item{...}{Additional arguments passed to \code{\link{nn_fit}}.}
}
\details{
\code{grid} should be a named list whose names are arguments accepted by
\code{\link{nn_fit}}. Atomic vectors are expanded as candidate values. Use
lists for arguments that are themselves vectors, such as \code{hidden}.

Each candidate is fitted with \code{validation_split} and scored on the held-out
validation rows when available. Metrics where larger values are better are
inferred automatically for accuracy-like scores, F1-like scores, and R-squared;
loss, log loss, RMSE, and MAE are minimized. Set \code{maximize} explicitly to
override this behavior.

When \code{seed} is supplied, candidate fits receive deterministic consecutive
seeds. The returned object keeps fitted candidate models, which is convenient
for inspection but can use memory for large grids.
}
\value{
A \code{neuralnetwork_tune} object, a list with:
\describe{
\item{\code{results}}{Candidate table ranked by score.}
\item{\code{best_model}}{The selected fitted model.}
\item{\code{best_params}}{One-row data frame of selected hyperparameters.}
\item{\code{models}}{List of fitted candidate models.}
\item{\code{maximize}}{Logical flag indicating score direction.}
}
}
\seealso{
\code{\link{neuralnetwork-metrics}}, \code{\link{nn_fit}},
\code{\link{nn_cv}}
}
\examples{
tuned <- nn_tune(Species ~ ., iris,
  grid = list(hidden = list(4, c(6, 3)), learning_rate = c(0.01)),
  epochs = 3, validation_split = 0.2, seed = 1, verbose = FALSE)
tuned$best_params
}
