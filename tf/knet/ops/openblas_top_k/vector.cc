/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "vector.h"

#include <string.h>
#include <assert.h>

#include <iomanip>
#include <cmath>
#include <iostream>

#include "matrix.h"

namespace fasttext {

Vector::Vector() {
  m_ = 0;
  data_ = nullptr;
}

Vector::Vector(int64_t m) {
  m_ = m;
  data_ = new real[m];
}

Vector& Vector::operator=(const Vector& source) {
  delete[] data_;
  m_ = source.m_;

  data_ = new real[m_];

  for (int i = 0; i < m_; ++i) {
    data_[i] = source.data_[i];
  }

  return *this;
}

Vector::~Vector() {
  delete[] data_;
}

int64_t Vector::size() const {
  return m_;
}

void Vector::zero() {
  memset(data_, 0, sizeof(real) * m_);
}

real Vector::norm() const {
  real sum = 0;
  for (int64_t i = 0; i < m_; i++) {
    sum += data_[i] * data_[i];
  }
  return std::sqrt(sum);
}

void Vector::mul(real a) {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] *= a;
  }
}

real Vector::dot(const Vector& target) {
  assert(m_ == target.m_);
  real sum = 0.0;
  for (int64_t i = 0; i < m_; i++) {
    sum += data_[i] * target.data_[i];
  }
  return sum;
}

void Vector::addVector(const Vector& source) {
  assert(m_ == source.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] += source.data_[i];
  }
}

void Vector::addVector(const Vector& source, real s) {
  assert(m_ == source.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] += s * source.data_[i];
  }
}

void Vector::addRow(const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += A.at(i, j);
  }
}

void Vector::addRow(const Matrix& A, int64_t i, real a) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += a * A.at(i, j);
  }
}

void Vector::mul(const Matrix& A, const Vector& vec) {
  assert(A.m_ == m_);
  assert(A.n_ == vec.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = A.dotRow(vec, i);
  }
}

int64_t Vector::argmax() {
  real max = data_[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < m_; i++) {
    if (data_[i] > max) {
      max = data_[i];
      argmax = i;
    }
  }
  return argmax;
}

real& Vector::operator[](int64_t i) {
  return data_[i];
}

const real& Vector::operator[](int64_t i) const {
  return data_[i];
}

void Vector::concat(const Vector& v, const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == v.m_ + A.n_);
  for (int64_t j = 0; j < v.m_; ++j) {
    data_[j] = v[j];
  }

  for (int64_t j = v.m_; j < m_; ++j) {
    data_[j] = A.at(i, j);
  }
}

void Vector::split(const Vector& left, const Vector& right) {
  assert(m_ == left.m_ + right.m_);
  for (int64_t i = 0; i < left.m_; ++i) {
    left.data_[i] = data_[i];
  }

  for (int64_t i = left.m_; i < m_; ++i) {
    right.data_[i - left.m_] = data_[i];
  }
}

std::ostream& operator<<(std::ostream& os, const Vector& v)
{
  os << std::setprecision(5);
  for (int64_t j = 0; j < v.m_; j++) {
    os << v.data_[j] << ' ';
  }
  return os;
}

void Vector::save(std::ostream& out) {
  out.write((char*) &m_, sizeof(int64_t));
  out.write((char*) data_, m_ * sizeof(real));
}

void Vector::load(std::istream& in) {
  in.read((char*) &m_, sizeof(int64_t));
  delete[] data_;
  data_ = new real[m_];
  in.read((char*) data_, m_ * sizeof(real));
}


}
