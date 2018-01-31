
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

     OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count_label", &args_->min_count_label));
     LOG(INFO) << "min_count_label: " << args_->min_count_label;

     OP_REQUIRES_OK(ctx, ctx->GetAttr("label", &args_->label));
     LOG(INFO) << "label: " << args_->label;
   }

   void PreProcessTrainData(OpKernelConstruction* ctx) {
     std::ifstream ifs(args_->train_data);
     OP_REQUIRES(ctx, ifs.is_open(), errors::Unavailable("File open failed."));
     dict_->readFromFile(ifs);
     ifs.close();
   }

   void Compute(OpKernelContext* ctx) override {
   }

 private:
   std::shared_ptr<::fasttext::Args> args_;
   std::shared_ptr<::fasttext::Dictionary> dict_;
};

REGISTER_KERNEL_BUILDER(Name("Fasttext").Device(DEVICE_CPU), FasttextOp);

}  // namespace tensorflow
