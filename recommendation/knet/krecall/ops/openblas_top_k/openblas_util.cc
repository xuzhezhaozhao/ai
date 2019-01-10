#include "openblas_util.h"
#include "real.h"

using fasttext::real;
using fasttext::Matrix;
using fasttext::Vector;

namespace {
void cblas_sgemv_helper(const real *a, const real *x, real *y, int m, int n) {
  real alpha = 1.0;
  int lda = m;
  int incx = 1;
  real beta = 0.0;
  int incy = 1;
  cblas_sgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta,
              y, incy);
}
}

void cblas_vec_dot_matrix(const Vector &vec, const Matrix &mat,
                          Vector &output) {
  assert(output.m_ == mat.m_);
  assert(mat.n_ == vec.m_);

  int m = static_cast<int>(mat.m_);
  int n = static_cast<int>(mat.n_);
  const real *a = mat.data_;
  const real *x = vec.data_;
  real *y = output.data_;
  cblas_sgemv_helper(a, x, y, m, n);
}


void cblas_vec_dot_matrix(const real *vec, int sz,
                          const fasttext::Matrix &mat,
                          fasttext::Vector &output) {
  assert(output.m_ == mat.m_);
  assert(mat.n_ == sz);

  int m = static_cast<int>(mat.m_);
  int n = static_cast<int>(mat.n_);
  const real *a = mat.data_;
  const real *x = vec;
  real *y = output.data_;
  cblas_sgemv_helper(a, x, y, m, n);
}
