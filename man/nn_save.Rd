\name{nn_save}
\alias{nn_save}
\alias{nn_load}
\title{Save and load neuralnetwork models}
\description{
Save a fitted \code{neuralnetwork} model to disk and load it back.
}
\usage{
nn_save(model, path, compress = TRUE)

nn_load(path)
}
\arguments{
\item{model}{A fitted \code{neuralnetwork} object.}
\item{path}{Path to an RDS file.}
\item{compress}{Compression argument passed to \code{\link[base]{saveRDS}}.}
}
\details{
The saved file contains the fitted model object, including preprocessing
metadata, fitted parameters, and training history. \code{nn_load()} checks that
the path exists, the model class, parameter dimensions, finite values and
preprocessing metadata before returning it. Older valid models may still be
loaded for prediction, but corrected numerical diagnostics and inference
metadata require refitting. \code{nn_save()} expects the destination directory to
already exist; it does not create folders.
Models with holdouts retain validation data; consider this before sharing an
RDS file. Structural validation does not make an untrusted serialized R
object safe to load.
}
\value{
\code{nn_save()} returns the path invisibly. \code{nn_load()} returns a
\code{neuralnetwork} model.
}
\examples{
fit <- nn_fit(mpg ~ wt + hp, mtcars, hidden = 4, epochs = 5,
              verbose = FALSE, seed = 1)
path <- tempfile(fileext = ".rds")
nn_save(fit, path)
loaded <- nn_load(path)
predict(loaded, mtcars[1:3, ])
}
