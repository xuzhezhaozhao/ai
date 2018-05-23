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

#include "args.h"
#include "dictionary.h"

namespace tensorflow {

class FasttextExampleGenerateOp : public OpKernel {
 public:
  explicit FasttextExampleGenerateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    args_ = std::make_shared<::fasttext::Args>();
    ParseArgs(ctx);
    rng_.seed(args_->seed);

    dict_ = std::make_shared<::fasttext::Dictionary>(args_);
    PreProcessTrainData(ctx);
  }

  void ParseArgs(OpKernelConstruction* ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("train_data_path", &args_->train_data_path));
    LOG(INFO) << "train_data_path: " << args_->train_data_path;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim", &args_->dim));
    LOG(INFO) << "dim: " << args_->dim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("maxn", &args_->maxn));
    LOG(INFO) << "maxn: " << args_->maxn;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("minn", &args_->minn));
    LOG(INFO) << "minn: " << args_->minn;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("word_ngrams", &args_->word_ngrams));
    LOG(INFO) << "word_ngrams: " << args_->word_ngrams;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("bucket", &args_->bucket));
    LOG(INFO) << "bucket: " << args_->bucket;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("ws", &args_->ws));
    LOG(INFO) << "ws: " << args_->ws;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &args_->min_count));
    LOG(INFO) << "min_count: " << args_->min_count;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("t", &args_->t));
    LOG(INFO) << "t: " << args_->t;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("verbose", &args_->verbose));
    LOG(INFO) << "verbose: " << args_->verbose;

    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("min_count_label", &args_->min_count_label));
    LOG(INFO) << "min_count_label: " << args_->min_count_label;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("label", &args_->label));
    LOG(INFO) << "label: " << args_->label;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &args_->seed));
    LOG(INFO) << "seed: " << args_->seed;
  }

  void PreProcessTrainData(OpKernelConstruction* ctx) {
    LOG(INFO) << "Preprocess train data beginning ...";
    std::ifstream ifs(args_->train_data_path);
    OP_REQUIRES(ctx, ifs.is_open(), errors::Unavailable("File open failed."));
    dict_->readFromFile(ifs);
    ifs.close();
    LOG(INFO) << "Preprocess train data done.";
  }

  inline int transform_id(int id) { return id + 1; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    auto input = input_tensor.flat<std::string>();

    std::vector<int32_t> words;
    std::vector<std::vector<int>> insts;
    for (int i = 0; i < input.size(); ++i) {
      words.clear();
      std::stringstream ss(input(i));
      int ntokens = dict_->getLine(ss, words, rng_);

      std::vector<int> bow;
      std::uniform_int_distribution<> uniform(1, args_->ws);
      for (int w = 1; w < words.size(); w++) {
        int32_t boundary = uniform(rng_);
        bow.clear();
        for (int c = -boundary; c < 0; c++) {
          if (c != 0 && w + c >= 0) {
            const std::vector<int>& ngrams = dict_->getSubwords(words[w + c]);
            bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
          }
        }
        bow.push_back(words[w]);  // add label
        insts.push_back(bow);
      }
    }

    // Create output tensors
    TensorShape records_shape;
    records_shape.AddDim(insts.size());
    records_shape.AddDim(args_->ws);
    Tensor* records_tensor = NULL;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, records_shape, &records_tensor));

    TensorShape labels_shape;
    labels_shape.AddDim(insts.size());
    labels_shape.AddDim(1);
    Tensor* labels_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, labels_shape, &labels_tensor));

    auto records = records_tensor->flat<int>();
    auto labels = labels_tensor->flat<int>();
    int labels_index = 0, records_index = 0;
    for (auto& inst : insts) {
      OP_REQUIRES(ctx, inst.size() > 0,
                  errors::InvalidArgument("inst size is 0"));

      labels(labels_index++) = transform_id(inst.back());

      for (int i = 0; i < inst.size() - 1; ++i) {
        records(records_index++) = transform_id(inst[i]);
      }
      // padding
      for (int i = inst.size() - 1; i < args_->ws; ++i) {
        records(records_index++) = PADDING_INDEX;
      }
    }
  }

 private:
  const int PADDING_INDEX = 0;
  std::shared_ptr<::fasttext::Args> args_;
  std::shared_ptr<::fasttext::Dictionary> dict_;

  std::minstd_rand rng_;
};

REGISTER_KERNEL_BUILDER(Name("FasttextExampleGenerate").Device(DEVICE_CPU),
                        FasttextExampleGenerateOp);

}  // namespace tensorflow
