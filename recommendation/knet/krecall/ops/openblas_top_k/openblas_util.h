#pragma once

#include <assert.h>
#include "tensorflow/core/lib/core/stringpiece.h"

#include "deps/openblas/include/cblas.h"
#include "matrix.h"
#include "vector.h"

void cblas_vec_dot_matrix(const fasttext::Vector &vec,
                          const fasttext::Matrix &mat,
                          fasttext::Vector &output);

void cblas_vec_dot_matrix(const fasttext::real *vec, int sz,
                          const fasttext::Matrix &mat,
                          fasttext::Vector &output);
