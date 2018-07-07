#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/posix/posix_file_system.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {
class FasttextNegativeSamplerOp : public OpKernel {
 public:
  explicit FasttextNegativeSamplerOp(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        num_true_(0),
        num_sampled_(0),
        range_max_(0),
        seed_(0),
        unique_(true),
        rng_(seed_),
        negpos_(0) {
    LOG(ERROR) << "Init FasttextNegativeSamplerOp ...";

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_true", &num_true_));
    LOG(ERROR) << "num_true = " << num_true_;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_sampled", &num_sampled_));
    LOG(ERROR) << "num_sampled = " << num_sampled_;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("unique", &unique_));
    LOG(ERROR) << "unique = " << unique_;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("range_max", &range_max_));
    LOG(ERROR) << "range_max = " << range_max_;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    LOG(ERROR) << "seed = " << seed_;
    rng_.seed(seed_);

    Tensor unigrams;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unigrams", &unigrams));
    InitTableNegatives(ctx, unigrams);

    LOG(ERROR) << "Init FasttextNegativeSamplerOp OK";
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& true_classes_tensor = ctx->input(0);
    const auto& true_classes_shape = true_classes_tensor.shape();
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(true_classes_shape),
                errors::InvalidArgument("true_classes expects a Matrix."));
    OP_REQUIRES(ctx, true_classes_shape.dim_size(1) == num_true_,
                errors::InvalidArgument(
                    "true_classes shape not matched [-1, num_true]."));
    auto maxtrix_true_classes = true_classes_tensor.matrix<int64>();

    // Create output tensors
    TensorShape sampled_candidates_shape;
    sampled_candidates_shape.AddDim(true_classes_shape.dim_size(0));
    sampled_candidates_shape.AddDim(num_sampled_);
    Tensor* sampled_candidates_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, sampled_candidates_shape,
                                             &sampled_candidates_tensor));
  }

 private:
  void InitTableNegatives(OpKernelConstruction* ctx, const Tensor& unigrams) {
    LOG(ERROR) << "InitTableNegatives ...";
    auto& unigrams_shape = unigrams.shape();
    auto flat_unigrams = unigrams.flat<int64>();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(unigrams_shape),
                errors::InvalidArgument("unigrams expects a Vector."));
    OP_REQUIRES(
        ctx, flat_unigrams.size() == range_max_,
        errors::InvalidArgument("unigrams and range_max size not matched."));

    float z = 0.0;
    for (size_t i = 0; i < flat_unigrams.size(); ++i) {
      z += pow(flat_unigrams(i), 0.5);
    }

    for (size_t i = 0; i < flat_unigrams.size(); ++i) {
      float c = pow(flat_unigrams(i), 0.5);
      for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; ++j) {
        negatives_.push_back(i);
      }
    }
    std::shuffle(negatives_.begin(), negatives_.end(), rng_);
    LOG(ERROR) << "InitTableNegatives OK.";
  }

  int getNegative(int32_t target) {
    int negative;
    do {
      negative = negatives_[negpos_];
      negpos_ = (negpos_ + 1) % negatives_.size();
    } while (target == negative);
    return negative;
  }

  static const int NEGATIVE_TABLE_SIZE = 10000000;

  int64 num_true_;
  int64 num_sampled_;
  int64 range_max_;
  int64 seed_ = 0;
  bool unique_;

  std::vector<int> negatives_;
  std::minstd_rand rng_;
  size_t negpos_;
};

REGISTER_KERNEL_BUILDER(Name("FasttextNegativeSamplerOp").Device(DEVICE_CPU),
                        FasttextNegativeSamplerOp);

}  // namespace tensorflow
