\name{print.neuralnetwork}
\alias{print.neuralnetwork}
\title{Print a neuralnetwork model}
\description{
Print a fitted \code{neuralnetwork} model with architecture, optimizer,
loss, backend, training progress, final training metrics, and validation
metrics when available. L-BFGS fits report function evaluations instead of
epochs.
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
