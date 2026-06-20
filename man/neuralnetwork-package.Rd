\name{neuralnetwork-package}
\alias{neuralnetwork-package}
\alias{neuralnetwork}
\docType{package}
\title{Compact neural networks for tabular R workflows}
\description{
\pkg{neuralnetwork} fits compact multilayer perceptrons for regression,
binary classification, and multiclass classification. It keeps ordinary R
interfaces such as formulas, data frames, matrices, S3 prediction methods, and
small model objects, while adding practical training tools such as modern
optimizers, validation splits, early stopping, robust regression, tuning,
cross-validation, and permutation importance.
}
\details{
The main entry point is \code{\link{nn_fit}}. A typical workflow is:

\enumerate{
\item Fit a model with \code{\link{nn_fit}}.
\item Predict with \code{\link{predict.neuralnetwork}}.
\item Evaluate with \code{\link{nn_evaluate}}.
\item Tune or cross-validate with \code{\link{nn_tune}} and
\code{\link{nn_cv}} when the problem needs it.
\item Inspect feature importance with \code{\link{nn_permutation_importance}}.
}

Automatic defaults are deliberately compact. \code{hidden = "auto"} chooses a
small task-aware hidden layer layout, \code{optimizer = "auto"} uses L-BFGS for
small deterministic tabular models and Adam for stochastic training features,
and \code{backend = "auto"} uses the Rcpp forward-pass backend when available.

For users coming from \pkg{nnet} or \pkg{neuralnet}, the package includes helper
functions such as \code{\link{nn_multinom}}, \code{\link{nn_class_ind}},
\code{\link{nn_compute}}, \code{\link{nn_generalized_weights}},
\code{\link{nn_hessian}}, and \code{\link{nn_confint}}.
}
\section{Main functions}{
\itemize{
\item \code{\link{nn_fit}}: fit a neural network.
\item \code{\link{nn_evaluate}}: compute task-aware metrics.
\item \code{\link{nn_tune}}: run a compact validation-set grid search.
\item \code{\link{nn_cv}}: run repeated k-fold cross-validation.
\item \code{\link{nn_permutation_importance}}: estimate feature importance.
\item \code{\link{nn_save}} and \code{\link{nn_load}}: save and load models.
}
}
\section{References}{
\itemize{
\item \code{\link{neuralnetwork-metrics}}: metric definitions and metric
selection.
\item \code{\link{neuralnetwork-callbacks}}: callback state and return values.
\item \code{\link{neuralnetwork-objects}}: fitted object structure.
}
}
\seealso{
\code{\link{nn_fit}}, \code{\link{nn_evaluate}}, \code{\link{nn_tune}},
\code{\link{nn_cv}}, \code{\link{nn_permutation_importance}}
}
\examples{
fit <- nn_fit(Species ~ ., iris, hidden = "auto", epochs = 5,
              validation_split = 0.2, seed = 1, verbose = FALSE)
fit
nn_evaluate(fit, iris)
}
