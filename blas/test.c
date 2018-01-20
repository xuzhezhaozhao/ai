
#include <stdlib.h>

#include "blas/sgemv.h"

#include <stdio.h>
#include <time.h>
#include <sys/time.h>

int main(int argc, char *argv[]) {
  int i;

  char trans[] = {'N'};  // don't transform
  integer m = 150000;       // M x N
  integer n = 100;
  real alpha = 1.0;
  real *a = (real *)malloc(m * n * sizeof(real));
  for (i = 0; i < m * n; ++i) {
    a[i] = i;
  }

  integer lda = m;
  real *x = (real *)malloc(n * sizeof(real));
  for (i = 0; i < n; ++i) {
    x[i] = i;
  }

  integer incx = 1;
  real beta = 0.0;
  real *y = (real *)malloc(m * sizeof(real));
  integer incy = 1;

  struct timeval stv;
  gettimeofday(&stv, NULL);
  long long start = stv.tv_sec * 1000 + stv.tv_usec / 1000;

  /*clock_t start = clock();*/
  int ret = sgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  /*clock_t end = clock();*/

  gettimeofday(&stv, NULL);
  long long end = stv.tv_sec * 1000 + stv.tv_usec / 1000;
  printf("Elapsed time: %llu ms.\n", end - start);

  for (i = 0; i < 10; ++i) {
    printf("%.2f ", y[i]);
  }
  printf("\n");
  printf("%d\n", ret);

  return 0;
}
