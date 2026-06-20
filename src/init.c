#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern SEXP neuralnetwork_dense_bias(SEXP, SEXP, SEXP);
extern SEXP neuralnetwork_activate(SEXP, SEXP);
extern SEXP neuralnetwork_sigmoid(SEXP);
extern SEXP neuralnetwork_softmax(SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"neuralnetwork_dense_bias", (DL_FUNC) &neuralnetwork_dense_bias, 3},
  {"neuralnetwork_activate", (DL_FUNC) &neuralnetwork_activate, 2},
  {"neuralnetwork_sigmoid", (DL_FUNC) &neuralnetwork_sigmoid, 1},
  {"neuralnetwork_softmax", (DL_FUNC) &neuralnetwork_softmax, 1},
  {NULL, NULL, 0}
};

void R_init_neuralnetwork(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
