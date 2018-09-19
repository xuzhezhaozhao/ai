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

#include "../../fe/feature_manager.h"

namespace tensorflow {
class KrankInputOp : public OpKernel {
 public:
  explicit KrankInputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), feature_manager_() {
    LOG(INFO) << "Init KrankInputOp ...";

    std::string fm_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_manager_path", &fm_path));
    feature_manager_.load(fm_path);

    LOG(INFO) << "Init KrankInputOp OK.";
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    auto flat_input = input_tensor.flat<std::string>();
    OP_REQUIRES(ctx, flat_input.size() == 1,
                errors::InvalidArgument("input size not 1"));
    const std::string& s = flat_input(0);

    fe::TransformedFeature feature = feature_manager_.transform(s);
  }

 private:
  fe::FeatureManager feature_manager_;
};

REGISTER_KERNEL_BUILDER(Name("KrankInput").Device(DEVICE_CPU), KrankInputOp);

}  // namespace tensorflow
