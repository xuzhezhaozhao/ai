#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/posix/posix_file_system.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include <time.h>

#include <fstream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "args.h"
#include "defines.h"
#include "dictionary.h"

namespace tensorflow {
class FasttextDictIdLookupOp : public OpKernel {
 public:
  explicit FasttextDictIdLookupOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    LOG(INFO) << "Init FasttextDictIdLookupOp ...";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dict_dir", &dict_dir_));
    LOG(INFO) << "dict_dir: " << dict_dir_;
    ParseDictWords(ctx);
    LOG(INFO) << "Init FasttextDictIdLookupOp OK";
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    auto input = input_tensor.flat<std::string>();

    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<int>();
    for (int i = 0; i < input.size(); ++i) {
      auto word = input(i);
      auto it = word2id_.find(word);
      if (it == word2id_.end()) {
        output(i) = PADDING_INDEX;
      } else {
        output(i) = it->second;
      }
    }
  }

 private:
  void ParseDictWords(OpKernelConstruction* ctx) {
    auto dict_words = ::tensorflow::io::JoinPath(dict_dir_, DICT_WORDS);
    LOG(INFO) << "Parse dict words from " << dict_words << " ...";

    std::ifstream ifs(dict_words);
    OP_REQUIRES(ctx, ifs.is_open(), errors::Unavailable("file open failed"));
    std::string line;
    int id = 1;  // id begin with 1 because of PADDING
    while (!ifs.eof()) {
      std::getline(ifs, line);
      if (line.empty()) {
        continue;
      }
      word2id_[line] = id++;
    }
    OP_REQUIRES(ctx, !ifs.bad(), errors::Unavailable("Read error"));
    OP_REQUIRES(ctx, word2id_.size() > 0,
                errors::Unavailable("Empty dict words file"));
    LOG(INFO) << "Dict words size = " << word2id_.size();
    LOG(INFO) << "Parse dict words OK";
  }

  std::unordered_map<std::string, int> word2id_;
  std::string dict_dir_;
};

REGISTER_KERNEL_BUILDER(Name("FasttextDictIdLookup").Device(DEVICE_CPU),
                        FasttextDictIdLookupOp);

}  // tensorflow
