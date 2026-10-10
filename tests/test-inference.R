library(neuralnetwork)
internal <- function(name) getFromNamespace(name, "neuralnetwork")
near <- function(a, b, tol = 1e-4) stopifnot(isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = tol)))
fails <- function(expr, pattern) {
  e <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(e, "error"), grepl(pattern, conditionMessage(e), fixed = TRUE))
}
set.seed(823)
d <- data.frame(x = seq(-2, 2, length.out = 120))
d$y <- 1.3 + 0.8 * d$x + rnorm(120, sd = 0.5)
for (backend in c("r", "rcpp")) {
  fit <- nn_fit(y ~ x, d, hidden = 0, optimizer = "lbfgs", scale = FALSE,
                y_scale = FALSE, epochs = 500, backend = backend, seed = 7, verbose = FALSE)
  ref <- lm(y ~ x, d)
  ci <- nn_confint(fit, d)
  near(ci$estimate, coef(ref)[c(2, 1)])
  near(ci$std.error, sqrt(diag(vcov(ref)))[c(2, 1)])
  near(ci$conf.low, confint(ref)[c(2, 1), 1])
  near(ci$conf.high, confint(ref)[c(2, 1), 2])
  gw <- nn_generalized_weights(fit, d)
  near(gw[, 1], rep(coef(ref)[2], nrow(d)))
  scaled <- nn_fit(y ~ x, d, hidden = 0, optimizer = "lbfgs", epochs = 500,
                   backend = backend, seed = 7, verbose = FALSE)
  near(nn_generalized_weights(scaled, d)[, 1], rep(coef(ref)[2], nrow(d)))
  h <- nn_hessian(fit, d)
  near(h, crossprod(cbind(d$x, 1)) / nrow(d))
  fails(nn_generalized_weights(fit, d, response = NA_real_), "response")
  fails(nn_generalized_weights(fit, d, response = 1.5), "response")
  fails(nn_generalized_weights(fit, d, response = c("y", "y")), "response")
  fails(internal(".resolve_feature_index")(fit, Inf), "Feature")
  fails(nn_confint(fit, d[-1, ]), "original training observations")
  old <- fit
  old$training_options <- NULL
  fails(nn_confint(old, d), "Parameter intervals require")

  d$cls <- factor(rbinom(120, 1, plogis(-0.3 + 0.7 * d$x)), levels = 0:1)
  logistic <- nn_fit(cls ~ x, d, hidden = 0, optimizer = "lbfgs", scale = FALSE,
                     epochs = 500, backend = backend, seed = 9, verbose = FALSE)
  ref <- glm(cls ~ x, d, family = binomial())
  ci <- nn_confint(logistic, d)
  near(ci$estimate, coef(ref)[c(2, 1)])
  near(ci$std.error, sqrt(diag(vcov(ref)))[c(2, 1)])
  near(ci$conf.low, coef(ref)[c(2, 1)] - qnorm(0.975) * sqrt(diag(vcov(ref)))[c(2, 1)])
  intercept <- nn_fit(cls ~ 1, d, hidden = 0, optimizer = "lbfgs", epochs = 300,
                      backend = backend, seed = 2, verbose = FALSE)
  ici <- nn_confint(intercept, d)
  near(ici$std.error, sqrt(1 / (120 * mean(d$cls == "1") * mean(d$cls == "0"))))
  stopifnot(nrow(ici) == 1L, length(internal(".flatten_params")(intercept$params)$names) == 1L)
  for (change in list(list(hidden = 2), list(l2 = 0.1), list(sample_weight = rep(2, 120)),
                      list(loss = "huber"), list(optimizer = "adam", epochs = 2))) {
    args <- utils::modifyList(list(x = y ~ x, data = d, hidden = 0, optimizer = "lbfgs",
                                  epochs = 200, seed = 2, backend = backend, verbose = FALSE), change)
    bad <- do.call(nn_fit, args)
    fails(nn_confint(bad, d), "Parameter intervals require")
  }
  singular <- nn_fit(y ~ x + I(2 * x), d, hidden = 0, optimizer = "lbfgs", epochs = 500,
                     backend = backend, seed = 3, verbose = FALSE)
  fails(nn_confint(singular, d), "singular or ill-conditioned")
  one <- nn_fit(matrix(0, 30, 0), matrix(0, 30, 1), task = "classification", hidden = 0,
                 optimizer = "lbfgs", epochs = 300, backend = backend, seed = 3, verbose = FALSE)
  fails(nn_confint(one, matrix(0, 30, 0), matrix(0, 30, 1)), "one-class or saturated")
}
cat("PASS: lm/glm estimates, standard errors, intervals, intercept-only fits and inference refusals on both backends.\n")
