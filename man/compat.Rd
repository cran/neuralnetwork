\name{neuralnetwork-compat}
\alias{neuralnetwork-compat}
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
Small helpers that cover common \code{nnet} and \code{neuralnet} tasks.
}
\usage{
nn_class_ind(classes)

nn_which_is_max(x)

nn_multinom(formula, data, weights = NULL, ...)

nn_compute(model, newdata)

nn_generalized_weights(model, newdata, response = 1L,
  epsilon = 1e-4)

nn_gwplot(model, newdata, selected.covariate = 1L,
  response = 1L, ...)

nn_hessian(model, newdata, y = NULL, epsilon = 1e-4,
  max_params = 120)

nn_confint(model, newdata, y = NULL, level = 0.95,
  epsilon = 1e-4, max_params = 120)
}
\arguments{
\item{classes}{A non-empty factor or vector of classes without missing values.}
\item{x}{A vector, matrix, or data frame.}
\item{formula, data, weights, ...}{Arguments passed to \code{\link{nn_fit}}.}
\item{model}{A fitted \code{neuralnetwork} model.}
\item{newdata}{New predictor data.}
\item{response}{Response name or index.}
\item{epsilon}{Positive finite-difference step for generalized weights and
Hessian calculations. Accepted but not used by the analytical interval
calculation.}
\item{selected.covariate}{Predictor name or index for plotting generalized weights.}
\item{y}{Optional truth vector or matrix, with one row or value per prediction
row.}
\item{level}{Confidence level between 0 and 1.}
\item{max_params}{Positive integer maximum parameter count allowed for
finite-difference Hessian.}
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
\item \code{nn_compute()} returns activations and final outputs in a
compute-style list.
\item \code{nn_generalized_weights()}: input sensitivities for a selected
response, estimated by central finite differences.
\item \code{nn_hessian()} computes finite-difference curvature of the mean
unweighted data loss, excluding regularization. It is a diagnostic, not a
covariance estimate or a substitute for total sample information.
\item \code{nn_confint()} provides parameter intervals only for converged,
unweighted, unregularized L-BFGS fits without hidden layers: single-output
squared-error regression or binary logistic classification. Older models
without training metadata must be refitted. It uses analytical total sample
information, estimated residual variance and Student t limits for regression,
and unit dispersion with normal limits for logistic models. Supply the
original training observations. Singular or ill-conditioned information is
an error; no ridge or negative-variance truncation is used.
}

Sensitivities are central finite differences per unit of an encoded input
before predictor scaling. Regression output is back-transformed from target
scaling. Classification sensitivities refer to probabilities, not log odds.
For \code{poly()}, interactions and factor contrasts, columns are encoded
features rather than original variables. These are not identical to
\pkg{neuralnet}'s generalized weights.
The sensitivity plot pads an almost constant vertical range to avoid
magnifying finite-difference noise; an explicit \code{ylim} overrides it.

Parameter intervals refer to internal weights and biases, including any
predictor or target scaling. Use \code{scale = FALSE, y_scale = FALSE} to
compare linear coefficients directly with \code{lm()} or \code{glm()}.
The regression intervals assume independent Gaussian errors with constant
variance; logistic intervals are asymptotic and can be unreliable with small
samples or separation. One-class outcomes or fitted probabilities within
\eqn{10^{-6}} of zero or one are refused for intervals. Hidden-layer weights
are not generally identifiable;
multiclass softmax also has a redundant parameterization. Neither receives
parameter intervals from this helper.
}
\value{
\code{nn_compute()} returns a list with \code{neurons} and \code{net.result}.

\code{nn_generalized_weights()} returns a matrix of input sensitivities.
\code{nn_hessian()} returns a curvature matrix.

\code{nn_confint()} returns a table of estimates, standard errors and limits.
}
\examples{
nn_class_ind(iris$Species[1:5])
fit <- nn_multinom(Species ~ ., iris, epochs = 5,
                   verbose = FALSE, seed = 1)
nn_compute(fit, iris[1:3, ])$net.result

linear <- nn_fit(mpg ~ wt, mtcars, hidden = 0, optimizer = "lbfgs",
                 scale = FALSE, y_scale = FALSE, epochs = 300,
                 verbose = FALSE, seed = 2)
nn_confint(linear, mtcars)
confint(lm(mpg ~ wt, mtcars)) # Same limits, in intercept-first order
}
