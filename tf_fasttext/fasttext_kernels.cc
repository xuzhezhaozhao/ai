
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

#include <fstream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "args.h"
#include "dictionary.h"

namespace tensorflow {

class FasttextOp : public OpKernel {
 public:
  explicit FasttextOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    args_ = std::make_shared<::fasttext::Args>();
    ParseArgs(ctx);

    dict_ = std::make_shared<::fasttext::Dictionary>(args_);
    PreProcessTrainData(ctx);
  }

  void ParseArgs(OpKernelConstruction* ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("train_data", &args_->train_data));
    LOG(INFO) << "train_data: " << args_->train_data;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("lr", &args_->lr));
    LOG(INFO) << "lr: " << args_->lr;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("lr_update_rate", &args_->lr_update_rate));
    LOG(INFO) << "lr_update_rate: " << args_->lr_update_rate;

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

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epoch", &args_->epoch));
    LOG(INFO) << "epoch: " << args_->epoch;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &args_->min_count));
    LOG(INFO) << "min_count: " << args_->min_count;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("neg", &args_->neg));
    LOG(INFO) << "neg: " << args_->neg;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("t", &args_->t));
    LOG(INFO) << "t: " << args_->t;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("verbose", &args_->verbose));
    LOG(INFO) << "verbose: " << args_->verbose;

    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("min_count_label", &args_->min_count_label));
    LOG(INFO) << "min_count_label: " << args_->min_count_label;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("label", &args_->label));
    LOG(INFO) << "label: " << args_->label;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &args_->batch_size));
    LOG(INFO) << "batch_size: " << args_->batch_size;
  }

  void PreProcessTrainData(OpKernelConstruction* ctx) {
    LOG(INFO) << "Preprocess train data beginning ...";
    ifs_.open(args_->train_data);
    OP_REQUIRES(ctx, ifs_.is_open(), errors::Unavailable("File open failed."));
    dict_->readFromFile(ifs_);

    // plus one for padding id
    Tensor word(DT_STRING, TensorShape({dict_->nwords() + 1}));
    Tensor freq(DT_INT32, TensorShape({dict_->nwords() + 1}));
    // xcbow 每次的输入历史记录不固定，但 Tensor 必须指定 shape, 所以
    // Tensor 指定固定 shape, 不足的用 padding id 补足
    word.flat<string>()(kPaddingId) = "</padding>";
    freq.flat<int32>()(kPaddingId) = 0;
    for (int i = 0; i < dict_->nwords(); ++i) {
      const string& w = dict_->getWord(i);
      int64_t cnt = dict_->getWordCount(i);
      auto id = i + 1;
      word.flat<string>()(id) = w;
      freq.flat<int32>()(id) = cnt;
    }
    word_ = word;
    freq_ = freq;

    ifs_.seekg(std::streampos(0));
    LOG(INFO) << "Preprocess train data done.";
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_words_processed(DT_INT64, TensorShape({}));
    Tensor examples(DT_INT32, TensorShape({args_->batch_size, args_->ws}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({args_->batch_size}));
    auto Tlabels = labels.flat<int32>();

    {
      // Generate batch_size examples
      mutex_lock l(mu_);
      std::uniform_int_distribution<> uniform(1, args_->ws);
      int example_index = 0;
      for (int i = 0; i < args_->batch_size; ++i) {
        if (next_pos_ >= line_.size()) {
          while (true) {
            total_words_processed_ += dict_->getLine(ifs_, line_, rng_);
            if (line_.size() >= 2) break;
          }
          next_pos_ = 1;
        }

        int boundary = uniform(rng_);
        bow_.clear();
        int w = next_pos_;
        for (int c = -boundary; c < 0; ++c) {
          if (w + c >= 0) {
            auto& ngrams = dict_->getSubwords(line_[w + c]);
            bow_.insert(bow_.end(), ngrams.cbegin(), ngrams.cend());
          }
        }
        // example: bow_ + padding
        // label: line_[w]
        for (int k = 0; k < bow_.size(); ++k) {
          Texamples(example_index++) = bow_[k];
        }
        for (int k = bow_.size(); k < args_->ws; ++k) {
          Texamples(example_index++) = kPaddingId;
        }
        Tlabels(i) = line_[w];

        ++next_pos_;
      }

      words_per_epoch.scalar<int64>()() = dict_->nwords();
      current_epoch.scalar<int32>()() = current_epoch_;
      total_words_processed.scalar<int64>()() = total_words_processed_;
    }  // end mutex_lock guard

    ctx->set_output(0, word_);
    ctx->set_output(1, freq_);
    ctx->set_output(2, words_per_epoch);
    ctx->set_output(3, current_epoch);
    ctx->set_output(4, total_words_processed);
    ctx->set_output(5, examples);
    ctx->set_output(6, labels);
  }

 private:
  static const int32 kPaddingId = 0;

  std::shared_ptr<::fasttext::Args> args_;
  std::shared_ptr<::fasttext::Dictionary> dict_;

  // 一次读取一行，每个词作为 label 生成训练数据, 一次生成 batch_size 个
  // examples, next_pos_ 为在 line_ 数组中下一个目标词索引
  std::ifstream ifs_;
  std::vector<int32_t> line_;
  std::vector<int32_t> bow_;  // bag of words
  // target index at least 1
  int next_pos_ = 1;

  // TODO seed this random engine
  std::minstd_rand rng_;

  Tensor word_;
  Tensor freq_;

  mutex mu_;
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int64 total_words_processed_ GUARDED_BY(mu_) = 0;
};

REGISTER_KERNEL_BUILDER(Name("Fasttext").Device(DEVICE_CPU), FasttextOp);

}  // namespace tensorflow
