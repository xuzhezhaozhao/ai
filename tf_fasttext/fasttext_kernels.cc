
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

class FasttextOp : public OpKernel {
 public:
  explicit FasttextOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    args_ = std::make_shared<::fasttext::Args>();
    ParseArgs(ctx);

    rng_.seed(args_->seed);
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

    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &args_->seed));
    LOG(INFO) << "seed: " << args_->seed;
  }

  void PreProcessTrainData(OpKernelConstruction* ctx) {
    LOG(INFO) << "Preprocess train data beginning ...";
    std::ifstream ifs(args_->train_data);
    dict_->readFromFile(ifs);
    ifs.close();

    total_tokens_ = dict_->ntokens();

    Tensor word(DT_STRING, TensorShape({dict_->nwords()}));
    Tensor freq(DT_INT32, TensorShape({dict_->nwords()}));
    for (int i = 0; i < dict_->nwords(); ++i) {
      const string& w = dict_->getWord(i);
      int64 cnt = dict_->getWordCount(i);
      auto id = i;
      word.flat<string>()(id) = w;
      freq.flat<int32>()(id) = cnt;
    }
    word_ = word;
    freq_ = freq;

    ifs_.open(args_->train_data);
    OP_REQUIRES(ctx, ifs_.is_open(), errors::Unavailable("File open failed."));
    LOG(INFO) << "Preprocess train data done.";
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_tokens_processed(DT_INT64, TensorShape({}));
    Tensor examples(DT_INT32, TensorShape({args_->batch_size, args_->ws}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({args_->batch_size, 1}));
    auto Tlabels = labels.flat<int32>();
    Tensor valid_lengths(DT_INT32, TensorShape({args_->batch_size}));
    auto Tvalid_lengths = valid_lengths.flat<int32>();
    {
      // Generate batch_size examples
      mutex_lock l(mu_);
      std::uniform_int_distribution<> uniform(1, args_->ws);
      for (int batch = 0; batch < args_->batch_size; ++batch) {
        if (next_pos_ >= line_.size()) {
          total_tokens_processed_ += dict_->getLine(ifs_, line_, rng_);
          current_epoch_ = total_tokens_processed_ / total_tokens_;
          while (line_.size() < 2) {
            total_tokens_processed_ += dict_->getLine(ifs_, line_, rng_);
            current_epoch_ = total_tokens_processed_ / total_tokens_;
          }
          next_pos_ = 1;
        }
        bow_.clear();
        int boundary = uniform(rng_);
        int w = next_pos_;
        for (int c = -boundary; c < 0; ++c) {
          if (w + c >= 0) {
            bow_.push_back(line_[w + c]);
          }
        }

        // example: bow_
        for (int k = 0; k < bow_.size(); ++k) {
          Texamples(batch * args_->ws + k) = bow_[k];
        }

        // label: line_[w]
        Tlabels(batch) = line_[w];
        Tvalid_lengths(batch) = static_cast<int32>(bow_.size());
        ++next_pos_;
      }

      words_per_epoch.scalar<int64>()() = dict_->ntokens();
      current_epoch.scalar<int32>()() = current_epoch_;
      total_tokens_processed.scalar<int64>()() = total_tokens_processed_;
    }  // end mutex_lock guard

    ctx->set_output(0, word_);
    ctx->set_output(1, freq_);
    ctx->set_output(2, words_per_epoch);
    ctx->set_output(3, current_epoch);
    ctx->set_output(4, total_tokens_processed);
    ctx->set_output(5, examples);
    ctx->set_output(6, labels);
    ctx->set_output(7, valid_lengths);
  }

 private:
  std::shared_ptr<::fasttext::Args> args_;
  std::shared_ptr<::fasttext::Dictionary> dict_;

  // 一次读取一行，每个词作为 label 生成训练数据, 一次生成 batch_size 个
  // examples, next_pos_ 为在 line_ 数组中下一个目标词索引
  std::ifstream ifs_;
  std::vector<int32_t> line_;
  std::vector<int32_t> bow_;  // bag of words
  // target index at least 1
  int next_pos_ = 1;

  int64 total_tokens_ = 0;

  std::minstd_rand rng_;

  Tensor word_;
  Tensor freq_;

  mutex mu_;
  int32 current_epoch_ GUARDED_BY(mu_) = 0;
  int64 total_tokens_processed_ GUARDED_BY(mu_) = 0;
};

REGISTER_KERNEL_BUILDER(Name("Fasttext").Device(DEVICE_CPU), FasttextOp);

class NegTrainWord2vecOp : public OpKernel {
 public:
  explicit NegTrainWord2vecOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    base_.Init(0, 0);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_negative_samples", &num_samples_));

    std::vector<int32> vocab_count;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_count", &vocab_count));

    std::vector<float> vocab_weights;
    vocab_weights.reserve(vocab_count.size());
    for (const auto& f : vocab_count) {
      float r = std::pow(static_cast<float>(f), 0.75f);
      vocab_weights.push_back(r);
    }
    sampler_ = new random::DistributionSampler(vocab_weights);
  }

  ~NegTrainWord2vecOp() { delete sampler_; }

  void Compute(OpKernelContext* ctx) override {
    Tensor w_in = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_in.shape()),
                errors::InvalidArgument("Must be a matrix"));
    Tensor w_out = ctx->mutable_input(1, false);
    OP_REQUIRES(ctx, w_in.shape() == w_out.shape(),
                errors::InvalidArgument("w_in.shape == w_out.shape"));
    const Tensor& examples = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(examples.shape()),
                errors::InvalidArgument("Must be a vector"));
    const Tensor& labels = ctx->input(3);
    OP_REQUIRES(ctx, examples.shape() == labels.shape(),
                errors::InvalidArgument("examples.shape == labels.shape"));
    const Tensor& learning_rate = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(learning_rate.shape()),
                errors::InvalidArgument("Must be a scalar"));

    auto Tw_in = w_in.matrix<float>();
    auto Tw_out = w_out.matrix<float>();
    auto Texamples = examples.flat<int32>();
    auto Tlabels = labels.flat<int32>();
    auto lr = learning_rate.scalar<float>()();
    const int64 vocab_size = w_in.dim_size(0);
    const int64 dims = w_in.dim_size(1);
    const int64 batch_size = examples.dim_size(0);
    OP_REQUIRES(ctx, vocab_size == sampler_->num(),
                errors::InvalidArgument("vocab_size mismatches: ", vocab_size,
                                        " vs. ", sampler_->num()));

    // Gradient accumulator for v_in.
    Tensor buf(DT_FLOAT, TensorShape({dims}));
    auto Tbuf = buf.flat<float>();

    // Scalar buffer to hold sigmoid(+/- dot).
    Tensor g_buf(DT_FLOAT, TensorShape({}));
    auto g = g_buf.scalar<float>();

    // The following loop needs 2 random 32-bit values per negative
    // sample.  We reserve 8 values per sample just in case the
    // underlying implementation changes.
    auto rnd = base_.ReserveSamples32(batch_size * num_samples_ * 8);
    random::SimplePhilox srnd(&rnd);

    for (int64 i = 0; i < batch_size; ++i) {
      const int32 example = Texamples(i);
      DCHECK(0 <= example && example < vocab_size) << example;
      const int32 label = Tlabels(i);
      DCHECK(0 <= label && label < vocab_size) << label;
      auto v_in = Tw_in.chip<0>(example);

      // Positive: example predicts label.
      //   forward: x = v_in' * v_out
      //            l = log(sigmoid(x))
      //   backward: dl/dx = g = sigmoid(-x)
      //             dl/d(v_in) = g * v_out'
      //             dl/d(v_out) = v_in' * g
      {
        auto v_out = Tw_out.chip<0>(label);
        auto dot = (v_in * v_out).sum();
        g = (dot.exp() + 1.f).inverse();
        Tbuf = v_out * (g() * lr);
        v_out += v_in * (g() * lr);
      }

      // Negative samples:
      //   forward: x = v_in' * v_sample
      //            l = log(sigmoid(-x))
      //   backward: dl/dx = g = -sigmoid(x)
      //             dl/d(v_in) = g * v_out'
      //             dl/d(v_out) = v_in' * g
      for (int j = 0; j < num_samples_; ++j) {
        const int sample = sampler_->Sample(&srnd);
        if (sample == label) continue;  // Skip.
        auto v_sample = Tw_out.chip<0>(sample);
        auto dot = (v_in * v_sample).sum();
        g = -((-dot).exp() + 1.f).inverse();
        Tbuf += v_sample * (g() * lr);
        v_sample += v_in * (g() * lr);
      }

      // Applies the gradient on v_in.
      v_in += Tbuf;
    }
  }

 private:
  int32 num_samples_ = 0;
  random::DistributionSampler* sampler_ = nullptr;
  GuardedPhiloxRandom base_;
};

REGISTER_KERNEL_BUILDER(Name("NegTrainWord2vec").Device(DEVICE_CPU),
                        NegTrainWord2vecOp);

}  // namespace tensorflow
