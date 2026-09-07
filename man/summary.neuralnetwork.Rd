\name{summary.neuralnetwork}
\alias{summary.neuralnetwork}
\alias{coef.neuralnetwork}
\title{Summarize neuralnetwork models}
\description{
Return model metadata and final training metrics, or extract raw fitted
parameters.
}
\usage{
\method{summary}{neuralnetwork}(object, ...)

\method{coef}{neuralnetwork}(object, ...)
}
\arguments{
\item{object}{A fitted \code{neuralnetwork} object.}
\item{...}{Unused.}
}
\details{
\code{summary()} is intended for a compact training report. For L-BFGS fits it
reports function evaluations and the convergence code returned by
\code{\link[stats]{optim}}, which helps distinguish a normal optimizer stop
from an iteration limit. Use \code{coef()} when you need the raw weight matrices
and bias vectors for inspection or custom post-processing.
}
\value{
\code{summary()} returns a summary object. \code{coef()} returns a list of
weight matrices and bias vectors.
}
