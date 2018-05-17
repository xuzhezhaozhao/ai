#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include <time.h>

#include <fstream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace tensorflow {

class StringSplitV2Op : public OpKernel {
 public:
   explicit StringSplitV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("delimter", &delimter_));
    LOG(INFO) << "delimter = " << delimter_;
   }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    auto input = input_tensor.flat<std::string>();

    size_t maxDim = 0;
    std::vector<std::vector<std::string>> results;
    for (int i = 0; i < input.size(); ++i) {
      results.push_back(tensorflow::str_util::Split(input(i), delimter_));
      maxDim = std::max(maxDim, results.back().size());
    }

    TensorShape shape = input_tensor.shape();
    shape.AddDim(maxDim);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output_tensor));
    auto output_flat = output_tensor->flat<std::string>();
    size_t index = 0;
    for (const auto& v : results) {
      for (const auto& e : v) {
        output_flat(index++) = e;
      }
      for (int i = 0; i < maxDim - v.size(); ++i) {
        output_flat(index++) = "";
      }
    }
  }

 private:
    std::string delimter_;
};

REGISTER_KERNEL_BUILDER(Name("StringSplitV2").Device(DEVICE_CPU),
                        StringSplitV2Op);

}  // namespace tensorflow
