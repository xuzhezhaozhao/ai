
#include <stdlib.h>

#include "cblas.h"

#include <stdio.h>
#include <time.h>
#include <sys/time.h>

typedef float real;

int main(int argc, char *argv[]) {
  int i;

  char trans[] = {'N'};  // don't transform
  int m = 250000;       // M x N
  int n = 100;
  real alpha = 1.0;
  real *a = (real *)malloc(m * n * sizeof(real));
  for (i = 0; i < m * n; ++i) {
    a[i] = i;
  }

  int lda = m;
  real *x = (real *)malloc(n * sizeof(real));
  for (i = 0; i < n; ++i) {
    x[i] = i;
  }

  int incx = 1;
  real beta = 0.0;
  real *y = (real *)malloc(m * sizeof(real));
  int incy = 1;

  struct timeval stv;
  gettimeofday(&stv, NULL);
  long long start = stv.tv_sec * 1000 + stv.tv_usec / 1000;

  /*int ret = sgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);*/
  openblas_set_num_threads(1);
  /*int nthread = openblas_get_num_threads();*/
  /*printf("nthread: %d\n", nthread);*/
  cblas_sgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);

  gettimeofday(&stv, NULL);
  long long end = stv.tv_sec * 1000 + stv.tv_usec / 1000;
  printf("Elapsed time: %llu ms.\n", end - start);

  for (i = 0; i < 10; ++i) {
    printf("%.2f ", y[i]);
  }
  printf("\n");

  return 0;
}
