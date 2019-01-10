/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_VECTOR_H
#define FASTTEXT_VECTOR_H

#include <cstdint>
#include <ostream>

#include "real.h"

namespace fasttext {

class Matrix;

class Vector {

  public:
    int64_t m_;
    real* data_;

    Vector();
    explicit Vector(int64_t);
    Vector& operator=(const Vector& source);

    Vector(const Vector&) = delete;

    ~Vector();

    real& operator[](int64_t);
    const real& operator[](int64_t) const;

    int64_t size() const;
    void zero();
    void mul(real);
    real dot(const Vector& target);
    real norm() const;
    void addVector(const Vector& source);
    void addVector(const Vector&, real);
    void addRow(const Matrix&, int64_t);
    void addRow(const Matrix&, int64_t, real);
    void mul(const Matrix&, const Vector&);
    int64_t argmax();

    void concat(const Vector&, const Matrix&, int64_t);
    void split(const Vector&, const Vector&);

    void save(std::ostream&);
    void load(std::istream&);
};

std::ostream& operator<<(std::ostream&, const Vector&);

}

#endif
