## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 4
)

## ----setup--------------------------------------------------------------------
library(neuralnetwork)

## ----classification-----------------------------------------------------------
fit_class <- nn_fit(
  Species ~ .,
  data = iris,
  hidden = "auto",
  optimizer = "auto",
  epochs = 10,
  validation_split = 0.2,
  seed = 1,
  verbose = FALSE
)

fit_class

## ----classification-predict---------------------------------------------------
predict(fit_class, iris[1:5, ], type = "class")
round(predict(fit_class, iris[1:5, ], type = "prob"), 3)

## ----classification-evaluate--------------------------------------------------
ev_class <- nn_evaluate(fit_class, iris)
ev_class

## ----binary-------------------------------------------------------------------
iris_binary <- subset(iris, Species != "virginica")
row_weight <- ifelse(iris_binary$Species == "versicolor", 1.5, 1)

fit_binary <- nn_fit(
  Species ~ .,
  data = iris_binary,
  hidden = c(6, 3),
  optimizer = "adam",
  epochs = 8,
  batch_size = 16,
  learning_rate = 0.01,
  sample_weight = row_weight,
  class_weight = "balanced",
  gradient_clip = 5,
  validation_split = 0.2,
  seed = 2,
  verbose = FALSE
)

round(predict(fit_binary, iris_binary[1:5, ], type = "prob"), 3)
nn_evaluate(fit_binary, iris_binary)

## ----regression---------------------------------------------------------------
fit_reg <- nn_fit(
  mpg ~ wt + hp + disp,
  data = mtcars,
  hidden = c(8, 4),
  optimizer = "adam",
  epochs = 25,
  batch_size = 8,
  learning_rate = 0.01,
  validation_split = 0.2,
  seed = 3,
  verbose = FALSE
)

fit_reg
round(predict(fit_reg, mtcars[1:5, ]), 2)
nn_evaluate(fit_reg, mtcars)

## ----huber--------------------------------------------------------------------
mtcars_outlier <- mtcars
mtcars_outlier$mpg[1] <- mtcars_outlier$mpg[1] + 40

fit_huber <- nn_fit(
  mpg ~ wt + hp,
  data = mtcars_outlier,
  hidden = 4,
  optimizer = "adam",
  loss = "huber",
  huber_delta = 1,
  epochs = 20,
  batch_size = 8,
  learning_rate = 0.01,
  seed = 4,
  verbose = FALSE
)

summary(fit_huber)

## ----controls-----------------------------------------------------------------
epochs_seen <- 0L

fit_callback <- nn_fit(
  mpg ~ wt + hp,
  data = mtcars,
  hidden = 4,
  optimizer = "adam",
  epochs = 20,
  batch_size = 8,
  learning_rate = 0.01,
  l2 = 1e-4,
  dropout = 0.05,
  gradient_clip = 5,
  validation_split = 0.2,
  callbacks = function(state) {
    epochs_seen <<- state$epoch
    if (state$epoch >= 2) {
      return(list(stop = TRUE))
    }
    NULL
  },
  seed = 5,
  verbose = FALSE
)

fit_callback

## ----tuning-------------------------------------------------------------------
tuned <- nn_tune(
  Species ~ .,
  data = iris,
  grid = list(
    hidden = list(4, c(6, 3)),
    learning_rate = c(0.01)
  ),
  metric = "balanced_accuracy",
  epochs = 4,
  validation_split = 0.2,
  seed = 6,
  verbose = FALSE
)

tuned
tuned$best_params

## ----cv-----------------------------------------------------------------------
cv <- nn_cv(
  Species ~ .,
  data = iris,
  k = 3,
  metric = "f1",
  hidden = 4,
  epochs = 2,
  seed = 7,
  verbose = FALSE
)

cv

## ----importance---------------------------------------------------------------
imp <- nn_permutation_importance(
  fit_reg,
  mtcars,
  metric = "mae",
  n_repeats = 2,
  seed = 8
)

imp

## ----save-load----------------------------------------------------------------
model_path <- tempfile(fileext = ".rds")
nn_save(fit_reg, model_path)
fit_loaded <- nn_load(model_path)

all.equal(
  predict(fit_reg, mtcars[1:3, ]),
  predict(fit_loaded, mtcars[1:3, ])
)

## ----compatibility------------------------------------------------------------
nn_class_ind(iris$Species[1:4])

computed <- nn_compute(fit_class, iris[1:2, ])
names(computed$neurons)
round(computed$net.result, 3)

