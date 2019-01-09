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

class CharCNNInputOp : public OpKernel {
 public:
  explicit CharCNNInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    LOG(INFO) << "Init  CharCNNInputOp ...";

    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_length", &max_length_));
    LOG(INFO) << "max_length: " << max_length_;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("label_str", &label_str_));
    LOG(INFO) << "label_str: " << label_str_;

    Tensor char_dict_tensor;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("char_dict", &char_dict_tensor));
    LoadDict(ctx, char_dict_tensor, char_dict_);
    LOG(INFO) << "char dict size = " << char_dict_.size();
    Tensor label_dict_tensor;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("label_dict", &label_dict_tensor));
    LoadDict(ctx, label_dict_tensor, label_dict_);
    LOG(INFO) << "label dict size = " << label_dict_.size();

    LOG(INFO) << "Init  CharCNNInputOp OK";
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    auto flat_input = input_tensor.flat<std::string>();
    OP_REQUIRES(ctx, flat_input.size() == 1,
                errors::InvalidArgument("input must be one string."));
    const std::string& s = flat_input(0);

    // Create output tensors
    Tensor* char_ids_tensor = NULL;
    Tensor* label_tensor = NULL;
    TensorShape shape;
    shape.AddDim(max_length_);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &char_ids_tensor));
    shape.Clear();
    shape.AddDim(1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, shape, &label_tensor));
    auto char_ids = char_ids_tensor->flat<int32>();
    auto label = label_tensor->flat<int64>();
    char_ids.setZero();
    label.setZero();

    if (s.empty()) {
      return;
    }

    int start = 0;
    if (s.find(label_str_) == 0) {
      // get label
      size_t pos = s.find_first_of(" \t");
      if (pos == std::string::npos) {
        pos = s.size();
      }
      auto it = label_dict_.find(s.substr(0, pos));
      if (it != label_dict_.end()) {
        label(0) = it->second;
      }
      start = pos + 1;
    }

    // read characters
    int cnt = 0;
    for (int j = start; j < s.size();) {
      std::string ch;
      // 处理 utf-8 字符
      if ((s[j] & 0xC0) == 0x80) continue;  // 跳过乱码的情况
      ch.push_back(s[j++]);
      while (j < s.size() && (s[j] & 0xC0) == 0x80) {
        ch.push_back(s[j++]);
      }
      auto it = char_dict_.find(ch);
      if (it != char_dict_.end()) {
        char_ids(cnt) = it->second;
        ++cnt;
        if (cnt == max_length_) {
          break;
        }
      }
    }
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

  std::unordered_map<std::string, int> char_dict_;
  std::unordered_map<std::string, int> label_dict_;
  std::string label_str_;
  int max_length_ = 0;
};

REGISTER_KERNEL_BUILDER(Name("CharCNNInput").Device(DEVICE_CPU),
                        CharCNNInputOp);

}  // namespace tensorflow
