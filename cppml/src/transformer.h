#ifndef _CPPML_TRANSFORMER_H_
#define _CPPML_TRANSFORMER_H_

#include <assert.h>
#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <iostream>

namespace cppml {

class TType {
 public:
  // default constructor
  TType() {
    type_ = INTEGER;
    value_.i64 = 0;
  }

  // implicit constructor
  TType(int i) {
    type_ = INTEGER;
    value_.i64 = i;
  }
  TType(int64_t i64) {
    type_ = INTEGER;
    value_.i64 = i64;
  }
  TType(float f) {
    type_ = NUMERIC;
    value_.d = f;
  }
  TType(double d) {
    type_ = NUMERIC;
    value_.d = d;
  }
  TType(const char* s) {
    type_ = STRING;
    value_.s = s;
  }

  TType(const std::string& s) {
    type_ = STRING;
    value_.s = s.data();
  }

  int64_t as_integer() {
    assert(type_ == INTEGER);
    return value_.i64;
  }
  double as_numeric() {
    assert(type_ == NUMERIC);
    return value_.d;
  }
  const char* as_string() {
    assert(type_ == STRING);
    return value_.s;
  }

 private:
  union {
    int64_t i64;
    double d;
    const char* s;
  } value_;

  enum { INTEGER = 0, NUMERIC, STRING } type_;
};

class Transformer {
 public:
  virtual void feed(TType t) = 0;
  virtual TType transform(TType t) const = 0;

  virtual void save(const std::string& filename) const = 0;
  virtual void load(const std::string& filename) = 0;
};

// Rescale features to [min, max], features outside the interval are clipped to
// the interval edges, which is also known as min-max normalization or
// Rescaling.
class MinMaxScaler : public Transformer {
 public:
  MinMaxScaler(double mmin = 0.0, double mmax = 1.0)
      : mmin_(mmin),
        mmax_(mmax),
        emin_(std::numeric_limits<double>::max() / 3.0),
        emax_(std::numeric_limits<double>::min() / 3.0) {
    assert(mmin <= mmax);
  }

  virtual void feed(TType t) {
    emax_ = std::max(emax_, t.as_numeric());
    emin_ = std::min(emin_, t.as_numeric());
  }

  virtual TType transform(TType t) const {
    if (emin_ == emax_) {
      return (mmin_ + mmax_) / 2.0;
    }
    double v = t.as_numeric();
    if (v < emin_) {
      return mmin_;
    }
    if (v > emax_) {
      return mmax_;
    }
    return (v - emin_) / (emax_ - emin_) * (mmax_ - mmin_) + mmin_;
  }

  double getMin() const { return emin_; }
  double getMax() const { return emax_; }

  // TODO(zhezhaoxu) implement
  virtual void save(const std::string& /* filename */) const {}
  virtual void load(const std::string& /* filename */) {}

 private:
  // user setting
  double mmin_;
  double mmax_;

  // from data
  double emin_;
  double emax_;
};

// TODO
class Imputer {};

// TODO
// Rescale each feature individually to range [-1, 1] by dividing through the
// largest maximum absolute value in each feature. It does not shift/center the
// data, and thus does not destroy any sparsity.
class MaxAbsScaler {};

// TODO
// QuantileDiscretizer takes a column with continuous features and outputs a
// column with binned categorical features.
class QuantileDiscretizer {};

// TODO
// Standardizes features by removing the mean and scaling to unit variance.
class StandardScaler {};

class MinCountStringIndexer {
 public:
  static constexpr int UNKNOWN_ID = 0;

  MinCountStringIndexer(int min_count = 0)
      : min_count_(min_count), feed_end_(false) {}

  virtual void feed(TType t) {
    assert(!feed_end_);
    ++counts_[t.as_string()];
  }

  // call this when you end up feeding data
  void feed_end() {
    for (auto p : counts_) {
      if (p.second >= min_count_) {
        table_[p.first] = strings_.size();
        strings_.push_back(p.first);
      }
    }
    counts_.clear();
    feed_end_ = true;
  }

  virtual TType transform(TType t) const {
    assert(feed_end_);

    auto it = table_.find(t.as_string());
    if (it == table_.end()) {
      return UNKNOWN_ID;
    }
    return it->second + 1;
  }

  int getMinCount() const { return min_count_; }
  virtual void save(std::ostream& out) const {
    int32_t sz = table_.size();
    assert(sz == (int32_t)strings_.size());
    out.write((char*) &sz, sizeof(int32_t));
    out.write((char*) &min_count_, sizeof(int32_t));
    out.write((char*) &feed_end_, sizeof(bool));
    for (auto& s : strings_) {
      out.write(s.data(), s.size() * sizeof(char));
      out.put(0);
    }
  }

  virtual void load(std::istream& in) {
    strings_.clear();
    table_.clear();
    counts_.clear();
    int32_t sz = 0;
    in.read((char*) &sz, sizeof(int32_t));
    in.read((char*) &min_count_, sizeof(int32_t));
    in.read((char*) &feed_end_, sizeof(bool));
    for (int32_t i = 0; i < sz; ++i) {
      char c;
      std::string s;
      while ((c = in.get()) != 0) {
        s.push_back(c);
      }
      strings_.push_back(s);
      table_[s] = i;
    }
  }

 private:
  int32_t min_count_;
  bool feed_end_;

  // TODO(zhezhaoxu) 先用简单 hash 表实现，上线后再考虑是否需要优化？（使用
  // vector 紧密存储）
  std::unordered_map<std::string, int> counts_;
  std::unordered_map<std::string, int> table_;

  // TODO use entry instead of string to store word count
  std::vector<std::string> strings_;
};

}  // namespace cppml

#endif
