\name{neuralnetwork-compat}
\alias{nn_class_ind}
\alias{nn_which_is_max}
\alias{nn_multinom}
\alias{nn_compute}
\alias{nn_generalized_weights}
\alias{nn_gwplot}
\alias{nn_hessian}
\alias{nn_confint}
\title{Compatibility helpers}
\description{
Small helpers that cover common \code{nnet} and \code{neuralnet} workflows.
}
\usage{
nn_class_ind(classes)

nn_which_is_max(x)

nn_multinom(formula, data, weights = NULL, ...)

nn_compute(model, newdata)

nn_generalized_weights(model, newdata, response = 1L, epsilon = 1e-4)

nn_gwplot(model, newdata, selected.covariate = 1L, response = 1L, ...)

nn_hessian(model, newdata, y = NULL, epsilon = 1e-4, max_params = 120)

nn_confint(model, newdata, y = NULL, level = 0.95,
  epsilon = 1e-4, max_params = 120)
}
\arguments{
\item{classes}{A factor or vector of classes.}
\item{x}{A vector, matrix, or data frame.}
\item{formula, data, weights, ...}{Arguments passed to \code{\link{nn_fit}}.}
\item{model}{A fitted \code{neuralnetwork} model.}
\item{newdata}{New predictor data.}
\item{response}{Response name or index.}
\item{epsilon}{Finite-difference step for generalized weights.}
\item{selected.covariate}{Predictor name or index for plotting generalized weights.}
\item{y}{Optional truth vector or matrix.}
\item{level}{Confidence level.}
\item{max_params}{Maximum parameter count allowed for finite-difference Hessian.}
}
\details{
These helpers are intentionally small wrappers around the fitted
\code{neuralnetwork} object.

\itemize{
\item \code{nn_class_ind()} creates a one-hot class indicator matrix.
\item \code{nn_which_is_max()} returns the first maximum index for a vector, or
row-wise maxima for matrices and data frames.
\item \code{nn_multinom()} fits a no-hidden-layer multiclass model, similar in
spirit to a multinomial log-linear model.
\item \code{nn_compute()} returns hidden-layer activations and final network
outputs, matching the compute-style workflow used by \pkg{neuralnet}.
\item \code{nn_generalized_weights()} estimates finite-difference input
sensitivities for a selected response.
\item \code{nn_hessian()} and \code{nn_confint()} use finite differences and are
intended for small models. Increase \code{max_params} only when the additional
computational cost is acceptable.
}
}
\value{
Depends on the helper. \code{nn_compute()} returns a list with \code{neurons}
and \code{net.result}. \code{nn_generalized_weights()} returns a matrix of
finite-difference sensitivities. \code{nn_hessian()} returns a matrix.
\code{nn_confint()} returns a data frame with estimates, standard errors, and
confidence limits.
}
\examples{
nn_class_ind(iris$Species[1:5])
fit <- nn_multinom(Species ~ ., iris, epochs = 5, verbose = FALSE, seed = 1)
nn_compute(fit, iris[1:3, ])$net.result
}
