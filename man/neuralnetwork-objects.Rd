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
\item{\code{task}}{One of these resolved tasks:\cr
\code{"regression"},\cr
\code{"binary_classification"}, or\cr
\code{"classification"}.}
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
\item{\code{training_data_loss}}{Selected model's unpenalized training data
loss, using combined training weights.}
\item{\code{training_units}}{Stored iteration units:\cr
L-BFGS: \code{"function_evaluations"}.\cr
Other optimizers: \code{"epochs"}.}
\item{\code{convergence}}{List with optimizer convergence metadata. For
L-BFGS fits, \code{code} and \code{message} come from \code{\link[stats]{optim}}.}
\item{\code{best_epoch}}{Epoch with the best monitored loss.}
\item{\code{validation}}{Prepared held-out predictors, encoded and raw
outcomes, sample weights and original row indices; \code{NULL} without a
holdout. Retaining these data increases model size.}
\item{\code{training_options}}{Regularization, dropout, weighting and
training-row-count metadata used to check inference eligibility.

The field \code{n} counts positive-weight training rows.

\code{excluded_zero_weight_rows} counts omitted training rows.}
\item{\code{stopped_early}}{Whether training stopped before the requested
number of epochs.}
}

The \code{history} data frame records each \code{epoch}, with:
\itemize{
\item Losses: \code{train_loss} and \code{validation_loss}.
\item Scores: \code{train_metric} and \code{validation_metric}.
\item Controls: \code{gradient_norm}, \code{learning_rate} and \code{backtracked}.
}
Validation columns are \code{NA} without a holdout.
Printed metrics and summaries describe the selected checkpoint, not the last
attempted epoch. The complete history still records later attempted epochs.

The object structure is documented for inspection. Code that only needs model
predictions should prefer the exported S3 methods and helper functions.
}
\seealso{
\code{\link{nn_fit}}, \code{\link{predict.neuralnetwork}},
\code{\link{summary.neuralnetwork}}, \code{\link{nn_evaluate}}
}
