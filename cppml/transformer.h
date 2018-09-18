#ifndef _CPPML_TRANSFORMER_H_
#define _CPPML_TRANSFORMER_H_

#include <algorithm>
#include <limits>
#include <string>

namespace cppml {

union TType {
  int i;
  long long ll;
  double d;
  char* s;
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
        emin_(std::numeric_limits<double>::max()),
        emax_(std::numeric_limits<double>::min()) {}

  virtual void feed(TType t) {
    emax_ = std::max(emax_, t.d);
    emin_ = std::min(emin_, t.d);
  }

  virtual TType transform(TType t) {
    TType ret;
    if (emin_ == emax_) {
      ret.d = (mmin_ + mmax_) / 2.0;
      return ret;
    }

    ret.d = (t.d - emin_) / (emax_ - emin_) * (mmax_ - mmin_) + mmin_;
    return ret;
  }

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
