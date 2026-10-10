\name{plot.neuralnetwork}
\alias{plot.neuralnetwork}
\title{Plot neuralnetwork training loss}
\description{
Plot training and optional validation loss from a fitted \code{neuralnetwork} model.
}
\details{
Both loss series determine the vertical range unless \code{ylim} is supplied.
A single stored diagnostic is drawn as a point. L-BFGS stores the final
diagnostic, not the full optimization trajectory.
}
\usage{
\method{plot}{neuralnetwork}(x, y = NULL, type = c("loss", "network"),
  main = "Training loss",
  xlab = "Epoch", ylab = "Loss", ...)
}
\arguments{
\item{x}{A fitted \code{neuralnetwork} object.}
\item{y}{Unused.}
\item{type}{Either \code{"loss"} for training curves or \code{"network"} for
a compact network diagram.}
\item{main, xlab, ylab}{Plot labels. Without an explicit \code{xlab}, L-BFGS
curves are labeled by function evaluation; other optimizers use epochs.}
\item{...}{Additional arguments passed to \code{\link[graphics]{plot}}.}
}
\value{
The input model, invisibly.
}
\examples{
fit <- nn_fit(mpg ~ wt + hp, mtcars, hidden = 4, epochs = 5,
              validation_split = 0.2, seed = 1, verbose = FALSE)
plot(fit)
plot(fit, type = "network")
}
