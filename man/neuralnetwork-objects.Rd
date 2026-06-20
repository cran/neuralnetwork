\name{neuralnetwork-objects}
\alias{neuralnetwork-objects}
\title{neuralnetwork model objects}
\description{
Structure of fitted model objects returned by \code{\link{nn_fit}}.
}
\details{
A fitted model is an S3 object of class \code{neuralnetwork}. Users usually
interact with it through \code{print()}, \code{predict()}, \code{plot()},
\code{summary()}, \code{coef()}, \code{\link{nn_evaluate}},
\code{\link{nn_save}}, and \code{\link{nn_load}}.

Important fields include:

\describe{
\item{\code{params}}{List of weight matrices and bias vectors.}
\item{\code{task}}{Resolved task: \code{"regression"},
\code{"binary_classification"}, or \code{"classification"}.}
\item{\code{hidden}}{Hidden-layer sizes.}
\item{\code{activation}}{Hidden-layer activation.}
\item{\code{optimizer}}{Resolved optimizer.}
\item{\code{loss}}{Resolved loss. Classification uses cross-entropy
internally.}
\item{\code{backend}}{Resolved forward-pass backend.}
\item{\code{scaler}}{Predictor centering and scaling information.}
\item{\code{target_scaler}}{Regression target scaling information.}
\item{\code{blueprint}}{Information used to prepare new data consistently.}
\item{\code{classes}}{Classification levels, or \code{NULL} for regression.}
\item{\code{history}}{Data frame with per-epoch training diagnostics.}
\item{\code{best_epoch}}{Epoch with the best monitored loss.}
\item{\code{stopped_early}}{Whether training stopped before the requested
number of epochs.}
}

The \code{history} data frame contains \code{epoch}, \code{train_loss},
\code{validation_loss}, \code{train_metric}, \code{validation_metric},
\code{gradient_norm}, \code{learning_rate}, and \code{backtracked}. Validation
columns are \code{NA} when no validation split was used.

The object structure is documented for inspection. Code that only needs model
predictions should prefer the exported S3 methods and helper functions.
}
\seealso{
\code{\link{nn_fit}}, \code{\link{predict.neuralnetwork}},
\code{\link{summary.neuralnetwork}}, \code{\link{nn_evaluate}}
}
