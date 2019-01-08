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

#include <unordered_map>

namespace tensorflow {

class TextCNNInputOp : public OpKernel {
 public:
  explicit TextCNNInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    LOG(INFO) << "Init  TextCNNInputOp ...";

    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_length", &max_length_));
    LOG(INFO) << "max_length: " << max_length_;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("log_per_lines", &log_per_lines_));
    LOG(INFO) << "log_per_lines: " << log_per_lines_;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_eval", &is_eval_));
    LOG(INFO) << "is_eval: " << is_eval_;

    Tensor word_dict_tensor;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("word_dict", &word_dict_tensor));
    LoadDict(ctx, word_dict_tensor, word_dict_);
    LOG(INFO) << "word dict size = " << word_dict_.size();
    Tensor label_dict_tensor;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("label_dict", &label_dict_tensor));
    LoadDict(ctx, label_dict_tensor, label_dict_);
    LOG(INFO) << "label dict size = " << label_dict_.size();

    LOG(INFO) << "Init  TextCNNInputOp OK";
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    auto flat_input = input_tensor.flat<std::string>();
    auto tokens = str_util::Split(flat_input(0), "\t ");

    // Create output tensors
    Tensor* word_ids_tensor = NULL;
    Tensor* label_tensor = NULL;
    TensorShape shape;
    shape.AddDim(max_length_);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &word_ids_tensor));
    shape.Clear();
    shape.AddDim(1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, shape, &label_tensor));
    auto word_ids = word_ids_tensor->flat<int32>();
    auto label = label_tensor->flat<int64>();
    word_ids.setZero();
    label.setZero();
    int cnt = 0;
    for (int i = 1; i < tokens.size(); ++i) {
      auto it = word_dict_.find(tokens[i]);
      if (it != word_dict_.end()) {
        word_ids(cnt) = it->second;
        ++cnt;
        if (cnt == max_length_) {
          break;
        }
      }
    }
    auto it = label_dict_.find(tokens[0]);
    label(0) = it->second;
  }

 private:
  void LoadDict(OpKernelConstruction* ctx, const Tensor& dict_tensor,
                std::unordered_map<std::string, int>& dict) {
    dict.clear();
    auto& dict_shape = dict_tensor.shape();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(dict_shape),
                errors::InvalidArgument("dict expects to be a Vector."));
    auto flat_dict = dict_tensor.flat<std::string>();
    for (int i = 0; i < flat_dict.size(); ++i) {
      auto p = dict.insert({flat_dict(i), i});
      OP_REQUIRES(
          ctx, p.second == true,
          errors::InvalidArgument("dict key duplicated, key = ", flat_dict(i)));
    }
  }

  std::unordered_map<std::string, int> word_dict_;
  std::unordered_map<std::string, int> label_dict_;
  int max_length_ = 0;
  int log_per_lines_ = 0;
  bool is_eval_ = false;
};

REGISTER_KERNEL_BUILDER(Name("TextCNNInput").Device(DEVICE_CPU),
                        TextCNNInputOp);

}  // namespace tensorflow
