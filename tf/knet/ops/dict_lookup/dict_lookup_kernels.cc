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

class DictLookupOp : public OpKernel {
 public:
  explicit DictLookupOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    LOG(ERROR) << "Init DictLookupOp ...";
    Tensor dict_tensor;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dict", &dict_tensor));
    LoadDict(ctx, dict_tensor);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_ws", &output_ws_));
    OP_REQUIRES(ctx, output_ws_ > 0,
                errors::InvalidArgument(
                    "ws should larger than 0, received ws = ", output_ws_));

    LOG(ERROR) << "Init DictLookupOp OK";
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const TensorShape& input_shape = input_tensor.shape();
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(input_shape),
                errors::InvalidArgument("input expects a Matrix."));
    auto input = input_tensor.matrix<std::string>();
    int batch_size = input_shape.dim_size(0);
    int input_ws = input_shape.dim_size(1);

    TensorShape ids_shape;
    TensorShape num_in_dict_shape;
    Tensor* ids_tensor = NULL;
    Tensor* num_in_dict_tensor = NULL;
    ids_shape.AddDim(batch_size);
    ids_shape.AddDim(output_ws_);
    num_in_dict_shape.AddDim(batch_size);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, ids_shape, &ids_tensor));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, num_in_dict_shape, &num_in_dict_tensor));

    auto ids = ids_tensor->matrix<int64>();
    auto flat_num_in_dict = num_in_dict_tensor->flat<int64>();
    for (int batch = 0; batch < batch_size; ++batch) {
      int num_in_dict = 0;
      for (int w = 0; w < input_ws; ++w) {
        auto& key = input(batch, w);
        if (IsInDict(key)) {
          ids(batch, num_in_dict) = mapping_[key];
          ++num_in_dict;
          if (num_in_dict >= output_ws_) {
            break;
          }
        }
      }
      // padding
      for (int i = num_in_dict; i < output_ws_; ++i) {
        ids(batch, i) = PADDING_INDEX;
      }
      flat_num_in_dict(batch) = num_in_dict;
    }
  }

 private:
  const int64 PADDING_INDEX = 0;

  void LoadDict(OpKernelConstruction* ctx, const Tensor& dict_tensor) {
    LOG(ERROR) << "Load dict ...";
    mapping_.clear();
    auto& dict_shape = dict_tensor.shape();
    auto flat_dict = dict_tensor.flat<std::string>();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(dict_shape),
                errors::InvalidArgument("dict expects a Vector."));
    for (int i = 0; i < flat_dict.size(); ++i) {
      auto p = mapping_.insert({flat_dict(i), i});
      OP_REQUIRES(
          ctx, p.second == true,
          errors::InvalidArgument("dict key duplicated, key = ", flat_dict(i)));
    }
    LOG(ERROR) << "Load dict OK, dict size = " << mapping_.size();
  }

  bool IsInDict(const std::string& key) {
    auto it = mapping_.find(key);
    if (it != mapping_.end() && it->second != PADDING_INDEX) {
      return true;
    }
    return false;
  }

  std::unordered_map<std::string, int64> mapping_;
  int64 output_ws_;
};

REGISTER_KERNEL_BUILDER(Name("DictLookup").Device(DEVICE_CPU), DictLookupOp);

}  // namespace tensorflow
