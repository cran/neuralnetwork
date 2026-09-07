\name{nn_fit}
\alias{nn_fit}
\title{Fit a small multilayer perceptron}
\description{
Fit a compact vectorized neural network for regression or classification.
}
\usage{
nn_fit(
  x,
  y = NULL,
  data = NULL,
  hidden = "auto",
  task = c("auto", "classification", "regression"),
  activation = c("auto", "relu", "tanh", "sigmoid", "leaky_relu"),
  optimizer = c("auto", "adam", "sgd", "momentum", "nesterov",
                "rprop", "grprop", "lbfgs"),
  backend = c("auto", "rcpp", "r"),
  loss = c("auto", "squared_error", "huber"),
  huber_delta = 1,
  epochs = 100,
  stepmax = NULL,
  batch_size = 32,
  learning_rate = 0.001,
  momentum = 0.9,
  beta1 = 0.9,
  beta2 = 0.999,
  epsilon = 1e-8,
  l2 = 0,
  dropout = 0,
  sample_weight = NULL,
  class_weight = NULL,
  gradient_clip = Inf,
  learning_rate_decay = 0,
  threshold = 0,
  validation_split = 0,
  patience = Inf,
  min_delta = 1e-4,
  scale = TRUE,
  y_scale = TRUE,
  shuffle = TRUE,
  seed = NULL,
  callbacks = NULL,
  verbose = TRUE
)
}
\arguments{
\item{x}{A formula, data frame, matrix, or numeric vector of predictors.}
\item{y}{Outcome vector or matrix. Required when \code{x} is not a formula.
For multiclass classification matrices, rows must be one-hot indicators or
class probabilities that sum to 1.}
\item{data}{Data frame used when \code{x} is a formula.}
\item{hidden}{Integer vector giving hidden layer sizes, \code{0} for no hidden
layer, or \code{"auto"} for a small default chosen from the task and input
width.}
\item{task}{One of \code{"auto"}, \code{"classification"}, or \code{"regression"}.}
\item{activation}{Hidden-layer activation. \code{"auto"} chooses a default
from the task and hidden-layer layout.}
\item{optimizer}{Optimizer, one of \code{"auto"}, \code{"adam"}, \code{"sgd"},
\code{"momentum"}, \code{"nesterov"}, \code{"rprop"}, \code{"grprop"}, or
\code{"lbfgs"}. \code{"auto"} uses
L-BFGS for small deterministic tabular models and Adam when stochastic
training features such as dropout or callbacks are used.}
\item{backend}{Forward-pass backend. \code{"auto"} uses Rcpp when available.}
\item{loss}{Regression loss. \code{"auto"} uses squared error; \code{"huber"}
uses a robust Huber loss. Classification uses cross-entropy internally.}
\item{huber_delta}{Positive finite Huber transition point, in the training
target scale when \code{y_scale = TRUE}.}
\item{epochs}{Positive integer number of training epochs.}
\item{stepmax}{Optional \pkg{neuralnet}-style alias for \code{epochs}.}
\item{batch_size}{Positive integer mini-batch size.}
\item{learning_rate}{Positive finite optimizer learning rate.}
\item{momentum}{Momentum coefficient in \code{[0, 1)} for \code{"momentum"} and
\code{"nesterov"} optimizers.}
\item{beta1, beta2}{Adam exponential-decay coefficients in \code{[0, 1)}.}
\item{epsilon}{Positive finite Adam denominator stabilizer.}
\item{l2}{Finite non-negative L2 regularization strength.}
\item{dropout}{Dropout probability. Either one value or one value per hidden layer.}
\item{sample_weight}{Optional non-negative row weights.}
\item{class_weight}{Optional classification weights. Use \code{"balanced"} or
a numeric vector with one value per class.}
\item{gradient_clip}{Global gradient-norm clip value. Use \code{Inf} to disable.}
\item{learning_rate_decay}{Finite non-negative per-epoch learning-rate decay.}
\item{threshold}{Finite non-negative gradient-norm convergence threshold. A
value of \code{0} disables this stopping rule.}
\item{validation_split}{Fraction of rows held out for validation.}
\item{patience}{Positive integer number of unimproved epochs before early
stopping, or \code{Inf} to disable patience-based stopping.}
\item{min_delta}{Finite non-negative minimum monitored-loss improvement.}
\item{scale}{Single \code{TRUE} or \code{FALSE}: whether to center and scale
predictors using the training split.}
\item{y_scale}{Single \code{TRUE} or \code{FALSE}: whether to center and scale
regression outcomes during training.}
\item{shuffle}{Single \code{TRUE} or \code{FALSE}: whether to shuffle rows each
epoch.}
\item{seed}{Optional non-negative integer random seed.}
\item{callbacks}{A function or list of functions called once per epoch. See
\code{\link{neuralnetwork-callbacks}}.}
\item{verbose}{Single \code{TRUE} or \code{FALSE}: whether to print progress
messages.}
}
\details{
\code{nn_fit()} accepts either a formula plus data, or explicit predictor and
outcome objects. Data-frame predictors are expanded with \code{model.matrix()},
so factors are encoded with the usual R contrast machinery. Matrix inputs are
used as supplied.

When \code{task = "auto"}, factor, character, and logical outcomes are treated
as classification. Numeric vectors and matrices are treated as regression.
Two-class classification uses a one-output sigmoid model internally, while
\code{predict(type = "prob")} still returns a two-column probability matrix.
Multiclass classification uses a softmax output. If \code{task} is set to
\code{"classification"} and \code{y} is a multi-column matrix, each row must be
a one-hot class indicator or a probability vector.

\code{hidden = "auto"} chooses a small architecture from the task and input
width. Use an integer vector such as \code{c(16, 8)} for multiple hidden
layers, or \code{0} for no hidden layer. \code{optimizer = "auto"} uses L-BFGS
for small deterministic
tabular models and Adam when stochastic training features such as dropout or
callbacks are used.

For L-BFGS fits, \code{epochs} and \code{stepmax} are passed to
\code{\link[stats]{optim}} as the iteration limit rather than interpreted as
mini-batch epochs. The printed model and \code{summary()} report function
evaluations for those fits.

For regression, \code{loss = "huber"} can be useful when a few observations are
unusually influential. If \code{y_scale = TRUE}, \code{huber_delta} is measured
on the scaled training target, not the original response scale.

Early stopping monitors validation loss when \code{validation_split > 0} and
training loss otherwise. The returned model stores the training history,
preprocessing information, fitted parameters, and enough blueprint information
to prepare new data consistently at prediction time.
}
\value{
An object of class \code{neuralnetwork}, a list containing fitted parameters,
preprocessing metadata, task information, training history, and S3 methods for
printing, prediction, plotting, summaries, and coefficients. See
\code{\link{neuralnetwork-objects}} for the object structure.
}
\seealso{
\code{\link{predict.neuralnetwork}}, \code{\link{nn_evaluate}},
\code{\link{nn_tune}}, \code{\link{nn_cv}},
\code{\link{neuralnetwork-callbacks}}, \code{\link{neuralnetwork-objects}}
}
\examples{
fit <- nn_fit(Species ~ ., data = iris, hidden = c(8, 4), epochs = 10,
              batch_size = 16, verbose = FALSE, seed = 1)
predict(fit, iris[1:3, ], type = "prob")
}
