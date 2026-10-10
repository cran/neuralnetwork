\name{neuralnetwork-callbacks}
\alias{neuralnetwork-callbacks}
\title{Callbacks in neuralnetwork training}
\description{
Reference for callback functions passed to \code{\link{nn_fit}} through the
\code{callbacks} argument.
}
\details{
Callbacks are called once at the end of each epoch for non-L-BFGS optimizers.
Pass either a single function or a list of functions. Each callback receives a
state list with:

\describe{
\item{\code{epoch}}{Current epoch.}
\item{\code{train_loss}}{Training loss after the epoch.}
\item{\code{validation_loss}}{Validation loss, or \code{NA} when no validation
split is used.}
\item{\code{train_metric}}{Training task metric, accuracy for classification or
RMSE for regression.}
\item{\code{validation_metric}}{Validation task metric, or \code{NA}.}
\item{\code{best_loss}}{Best monitored loss so far.}
\item{\code{best_epoch}}{Epoch with the best monitored loss.}
\item{\code{gradient_norm}}{Largest batch gradient norm observed in the epoch.}
\item{\code{learning_rate}}{Learning rate for the next epoch unless changed by
a callback.}
\item{\code{task}}{Resolved task name.}
}

A callback may return \code{FALSE} or \code{list(stop = TRUE)} to stop
training. Return \code{NULL} to leave training unchanged.

To change the next epoch's learning rate, return
\code{list(learning_rate = value)}. The value must be positive and finite.
If \code{stop} is returned in a list, it must be a single \code{TRUE} or
\code{FALSE}.

When multiple callbacks are supplied, they are run in order. If one callback
requests stopping, later callbacks are skipped for that epoch.
}
\seealso{
\code{\link{nn_fit}}
}
\examples{
halve_after_two <- function(state) {
  if (state$epoch == 2) {
    return(list(learning_rate = state$learning_rate * 0.5))
  }
  NULL
}

fit <- nn_fit(mpg ~ wt + hp, mtcars, hidden = 4, epochs = 4,
              callbacks = halve_after_two, seed = 1, verbose = FALSE)
fit
}
