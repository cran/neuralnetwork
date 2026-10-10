\name{nn_tune}
\alias{nn_tune}
\title{Tune neuralnetwork hyperparameters}
\description{
Fit a grid of \code{nn_fit()} candidates and rank them by validation
performance.
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
  error_action = c("stop", "continue"),
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
\item{error_action}{What to do when a candidate fails. \code{"stop"} raises the
first error. \code{"continue"} records the candidate error and ranks the
remaining usable candidates.}
\item{seed}{Optional non-negative integer random seed. Candidate fits receive
deterministic consecutive seeds derived from this value.}
\item{verbose}{Whether individual fits should print progress.}
\item{...}{Additional arguments passed to \code{\link{nn_fit}}.}
}
\details{
\code{grid} should be a named list whose names are arguments accepted by
\code{\link{nn_fit}}. Atomic vectors are expanded as candidate values. Use
lists for arguments that are themselves vectors, such as \code{hidden}.

All candidates share one validation row set drawn before fitting. Candidate
seeds vary initialization and training, not the comparison data. Data, seeds
and validation-row arguments cannot be grid entries. Grid values override
the same argument in \code{...}. Candidates must use a common task and
metric, and only finite scores can rank. Larger values are better for accuracy,
balanced accuracy, F1, macro F1, and R-squared. Loss, log loss, RMSE, and MAE
are minimized. Set \code{maximize} explicitly to
override this behavior.
Scoring sample weights are supplied through \code{...}, not the grid.
When ranking by \code{loss}, candidates must share the loss definition,
Huber delta (if used) and target scaler. To compare different losses, choose
a common reporting metric such as RMSE. Without a holdout, loss is training
data loss without the regularization penalty; it is not an estimate of
generalization error.

When \code{seed} is supplied, candidate fits receive deterministic consecutive
seeds. The returned object keeps fitted candidate models; large grids can use a
noticeable amount of memory.

By default, candidate errors stop the search. Use
\code{error_action = "continue"} for wider grids where some candidate
combinations may be invalid. Failed candidates and candidates without
an available score are kept in the results table with \code{success = FALSE},
an \code{error} message, and \code{NA} score/rank values.
}
\value{
A \code{neuralnetwork_tune} object, a list with:
\describe{
\item{\code{results}}{Ranked candidate table, with success and error columns.

The \code{candidate_id} column indexes \code{models}, even after sorting.}
\item{\code{best_model}}{The selected fitted model.}
\item{\code{best_params}}{One-row data frame of selected hyperparameters.}
\item{\code{models}}{List of fitted candidate models.}
\item{\code{errors}}{Character vector of candidate error messages.}
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
