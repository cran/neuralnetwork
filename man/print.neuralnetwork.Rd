\name{print.neuralnetwork}
\alias{print.neuralnetwork}
\title{Print a neuralnetwork model}
\description{
Print model architecture, optimizer, loss and backend.

Selected-checkpoint training metrics and available validation metrics follow
the training progress. L-BFGS reports function evaluations instead of epochs.
}
\usage{
\method{print}{neuralnetwork}(x, ...)
}
\arguments{
\item{x}{A fitted \code{neuralnetwork} object.}
\item{...}{Unused.}
}
\value{
The input model, invisibly.
}
