/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "matrix.h"

#include <assert.h>

#include <random>

#include "vector.h"

namespace fasttext {

Matrix::Matrix() {
  m_ = 0;
  n_ = 0;
  data_ = nullptr;
}

Matrix::Matrix(int64_t m, int64_t n) {
  m_ = m;
  n_ = n;
  data_ = new real[m * n];
}

Matrix::Matrix(const Matrix& other) {
  m_ = other.m_;
  n_ = other.n_;
  data_ = new real[m_ * n_];
  for (int64_t i = 0; i < (m_ * n_); i++) {
    data_[i] = other.data_[i];
  }
}

Matrix& Matrix::operator=(const Matrix& other) {
  Matrix temp(other);
  m_ = temp.m_;
  n_ = temp.n_;
  std::swap(data_, temp.data_);
  return *this;
}

Matrix::~Matrix() {
  delete[] data_;
}

void Matrix::zero() {
  for (int64_t i = 0; i < (m_ * n_); i++) {
      data_[i] = 0.0;
  }
}

void Matrix::uniform(real a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<real> uniform(-a, a);
  for (int64_t i = 0; i < (m_ * n_); i++) {
    data_[i] = uniform(rng);
  }
}

real Matrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real d = 0.0;
  for (int64_t j = 0; j < n_; j++) {
    d += at(i, j) * vec.data_[j];
  }
  return d;
}

void Matrix::addRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec.data_[j];
  }
}

void Matrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
  if (ie == -1) {ie = m_;}
  assert(ie <= nums.size());
  for (auto i = ib; i < ie; i++) {
    real n = nums[i-ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) *= n;
      }
    }
  }
}

void Matrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
  if (ie == -1) {ie = m_;}
  assert(ie <= denoms.size());
  for (auto i = ib; i < ie; i++) {
    real n = denoms[i-ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) /= n;
      }
    }
  }
}

real Matrix::l2NormRow(int64_t i) const {
  auto norm = 0.0;
  for (auto j = 0; j < n_; j++) {
    const real v = at(i,j);
    norm += v * v;
  }
  return std::sqrt(norm);
}

void Matrix::l2NormRow(Vector& norms) const {
  assert(norms.size() == m_);
    for (auto i = 0; i < m_; i++) {
      norms[i] = l2NormRow(i);
    }
}

void Matrix::save(std::ostream& out) {
  out.write((char*) &m_, sizeof(int64_t));
  out.write((char*) &n_, sizeof(int64_t));
  out.write((char*) data_, m_ * n_ * sizeof(real));
}

void Matrix::load(std::istream& in) {
  in.read((char*) &m_, sizeof(int64_t));
  in.read((char*) &n_, sizeof(int64_t));
  delete[] data_;
  data_ = new real[m_ * n_];
  in.read((char*) data_, m_ * n_ * sizeof(real));
}

void Matrix::shrinkSubset(const std::vector<int32_t> &subset) {
  int64_t new_m = subset.size();

  real *new_data = new real[new_m * n_];
  for (size_t k = 0; k < subset.size(); ++k) {
    int idx = subset[k];
    for (int i = 0; i < n_; ++i) {
      new_data[k * n_ + i] = data_[idx * n_ + i];
    }
  }

  delete[] data_;
  m_ = new_m;
  data_ = new_data;
}

void Matrix::convertColMajor() {
  real *new_data = new real[m_ * n_];
  // 遍历列
  int i = 0;
  for (int col = 0; col < n_; ++col) {
    for (int row = 0; row < m_; ++row) {
      new_data[i++] = data_[row * n_ + col];
    }
  }

  delete[] data_;
  data_ = new_data;
}

}
