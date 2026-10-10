\name{predict.neuralnetwork}
\alias{predict.neuralnetwork}
\title{Predict from a neuralnetwork model}
\description{
Predict classes, class probabilities, or numeric responses from a fitted
\code{neuralnetwork} model.
}
\usage{
\method{predict}{neuralnetwork}(object, newdata,
  type = c("response", "class", "prob"), ...)
}
\arguments{
\item{object}{A fitted \code{neuralnetwork} object.}
\item{newdata}{New predictor data. Formula and data-frame fits require the
same predictor variables used during fitting, with compatible factor levels.}
\item{type}{Prediction type. \code{"response"} returns class labels for
classification and numeric predictions for regression.}
\item{...}{Unused.}
}
\details{
For classification models, \code{type = "response"} and \code{type = "class"}
return class labels. \code{type = "prob"} returns a probability matrix with one
column per class. For regression models, \code{type = "response"} returns
numeric predictions on the original response scale. \code{type = "class"} and
\code{type = "prob"} are not valid for regression.

Target scaling is undone, but a formula transformation is not inverted:
\code{log(y) ~ x} predicts log-y, not raw y. Named matrix predictors are
matched by fitted column names; missing names are an error. Unnamed matrices
must have the fitted width and column order. Stored formula bases and
contrasts are reused even if current contrast options differ.
}
\value{
A factor for classification responses, a probability matrix for
\code{type = "prob"}, or numeric predictions for regression.
}
\examples{
fit <- nn_fit(Species ~ ., iris, hidden = 4, epochs = 5,
              seed = 1, verbose = FALSE)
predict(fit, iris[1:3, ], type = "class")
predict(fit, iris[1:3, ], type = "prob")
}
