#include <Rcpp.h>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <string>

using namespace Rcpp;

inline double stable_sigmoid_value(double x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + std::exp(-x));
  }
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

extern "C" SEXP neuralnetwork_dense_bias(SEXP xSEXP, SEXP wSEXP, SEXP bSEXP) {
BEGIN_RCPP
  NumericMatrix x(xSEXP);
  NumericMatrix w(wSEXP);
  NumericVector b(bSEXP);

  const int n = x.nrow();
  const int p = x.ncol();
  const int q = w.ncol();

  if (w.nrow() != p) {
    stop("Dense layer dimension mismatch.");
  }
  if (b.size() != q) {
    stop("Bias length does not match dense layer output width.");
  }

  NumericMatrix out(n, q);
  const double *x_ptr = REAL(x);
  const double *w_ptr = REAL(w);
  double *out_ptr = REAL(out);

  for (int j = 0; j < q; ++j) {
    double *out_col = out_ptr + static_cast<std::size_t>(n) * j;
    std::fill(out_col, out_col + n, b[j]);

    const double *w_col = w_ptr + static_cast<std::size_t>(p) * j;
    for (int k = 0; k < p; ++k) {
      const double weight = w_col[k];
      if (weight == 0.0) {
        continue;
      }
      const double *x_col = x_ptr + static_cast<std::size_t>(n) * k;
      for (int i = 0; i < n; ++i) {
        out_col[i] += x_col[i] * weight;
      }
    }
  }

  return out;
END_RCPP
}

extern "C" SEXP neuralnetwork_activate(SEXP xSEXP, SEXP activationSEXP) {
BEGIN_RCPP
  NumericMatrix x(xSEXP);
  const std::string activation = as<std::string>(activationSEXP);
  const int n = x.nrow();
  const int p = x.ncol();
  NumericMatrix out(n, p);

  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < n; ++i) {
      const double v = x(i, j);
      if (activation == "relu") {
        out(i, j) = v > 0.0 ? v : 0.0;
      } else if (activation == "leaky_relu") {
        out(i, j) = v > 0.0 ? v : 0.01 * v;
      } else if (activation == "tanh") {
        out(i, j) = std::tanh(v);
      } else if (activation == "sigmoid") {
        out(i, j) = stable_sigmoid_value(v);
      } else {
        stop("Unknown activation.");
      }
    }
  }

  return out;
END_RCPP
}

extern "C" SEXP neuralnetwork_sigmoid(SEXP xSEXP) {
BEGIN_RCPP
  NumericMatrix x(xSEXP);
  const int n = x.nrow();
  const int p = x.ncol();
  NumericMatrix out(n, p);

  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < n; ++i) {
      out(i, j) = stable_sigmoid_value(x(i, j));
    }
  }

  return out;
END_RCPP
}

extern "C" SEXP neuralnetwork_softmax(SEXP xSEXP) {
BEGIN_RCPP
  NumericMatrix x(xSEXP);
  const int n = x.nrow();
  const int p = x.ncol();
  NumericMatrix out(n, p);

  for (int i = 0; i < n; ++i) {
    double row_max = x(i, 0);
    for (int j = 1; j < p; ++j) {
      if (x(i, j) > row_max) {
        row_max = x(i, j);
      }
    }

    double total = 0.0;
    for (int j = 0; j < p; ++j) {
      const double value = std::exp(x(i, j) - row_max);
      out(i, j) = value;
      total += value;
    }

    for (int j = 0; j < p; ++j) {
      out(i, j) /= total;
    }
  }

  return out;
END_RCPP
}
