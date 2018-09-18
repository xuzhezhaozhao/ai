#ifndef _CPPML_TRANSFORMER_H_
#define _CPPML_TRANSFORMER_H_

#include <algorithm>
#include <limits>
#include <string>
#include <assert.h>

namespace cppml {

class TType {
 public:
  // default constructor
  TType() {
    type_ = INT32;
    value_.i32 = 0;
  }

  // implicit constructor
  TType(int32_t i32) {
    type_ = INT32;
    value_.i32 = i32;
  }
  TType(int64_t i64) {
    type_ = INT64;
    value_.i64 = i64;
  }
  TType(float f) {
    type_ = FLOAT;
    value_.f = f;
  }
  TType(double d) {
    type_ = DOUBLE;
    value_.d = d;
  }
  TType(char* s) {
    type_ = STRING;
    value_.s = s;
  }

  int32_t as_int32() {
    assert(type_ == INT32);
    return value_.i32;
  }
  int64_t as_int64() {
    assert(type_ == INT64);
    return value_.i64;
  }
  float as_float() {
    assert(type_ == FLOAT);
    return value_.f;
  }
  double as_double() {
    assert(type_ == DOUBLE);
    return value_.d;
  }
  char* as_string() {
    assert(type_ == STRING);
    return value_.s;
  }

 private:
  union {
    int32_t i32;
    int64_t i64;
    float f;
    double d;
    char* s;
  } value_;

  enum {
    INT32 = 0,
    INT64,
    FLOAT,
    DOUBLE,
    STRING
  } type_;
};

class Transformer {
 public:
  virtual void feed(TType t) = 0;
  virtual TType transform(TType t) = 0;

  virtual void save(const std::string& filename) = 0;
  virtual void load(const std::string& filename) = 0;
};

// see:
// https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.MinMaxScaler
// scale feature to [min, max]
class MinMaxScaler : public Transformer {
 public:
  MinMaxScaler(double mmin, double mmax)
      : mmin_(mmin),
        mmax_(mmax),
        emin_(std::numeric_limits<double>::max()/2.0),
        emax_(std::numeric_limits<double>::min()/2.0) {}

  virtual void feed(TType t) {
    emax_ = std::max(emax_, t.as_double());
    emin_ = std::min(emin_, t.as_double());
  }

  virtual TType transform(TType t) {
    TType ret;
    if (emin_ == emax_) {
      ret = (mmin_ + mmax_) / 2.0;
      return ret;
    }

    ret = (t.as_double() - emin_) / (emax_ - emin_) * (mmax_ - mmin_) + mmin_;
    return ret;
  }

  virtual void save(const std::string& filename) {}

  virtual void load(const std::string& filename) {}

 private:
  // user setting
  double mmin_;
  double mmax_;

  // from data
  double emin_;
  double emax_;
};

}  // namespace cppml

#endif
