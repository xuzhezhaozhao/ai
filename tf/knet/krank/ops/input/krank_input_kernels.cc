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

#include <time.h>

#include "../../fe/feature_manager.h"

namespace tensorflow {
class KrankInputOp : public OpKernel {
 public:
  explicit KrankInputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        global_lines_(0),
        num_label_level_1(0),
        num_label_level_2(0),
        num_label_level_3(0),
        num_label_level_4(0),
        num_label_level_5(0),
        feature_manager_(),
        ws_(5),
        num_evaluate_target_per_line_(1),
        log_per_lines_(10000),
        is_eval_(false) {
    LOG(INFO) << "Init KrankInputOp ...";

    rng_.seed(time(NULL));

    std::string fm_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_manager_path", &fm_path));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ws", &ws_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_evaluate_target_per_line",
                                     &num_evaluate_target_per_line_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("log_per_lines", &log_per_lines_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_eval", &is_eval_));
    feature_manager_.load(fm_path);

    LOG(INFO) << "Init KrankInputOp OK.";
  }

  void Compute(OpKernelContext* ctx) override {
    CalcStatistic();

    const Tensor& input_tensor = ctx->input(0);
    auto flat_input = input_tensor.flat<std::string>();
    OP_REQUIRES(ctx, flat_input.size() == 1,
                errors::InvalidArgument("input size not 1"));
    const std::string& s = flat_input(0);

    fe::TransformedFeature feature = feature_manager_.transform(s);

    std::vector<std::vector<int>> positive_records;
    std::vector<std::vector<int>> negative_records;
    std::vector<int64> targets;
    std::vector<int64> first_videos;
    std::vector<float> labels;
    std::uniform_int_distribution<> uniform(1, ws_);

    int w = 1;
    if (is_eval_) {
      w = std::max(w, static_cast<int>(feature.actions.size()) -
                          num_evaluate_target_per_line_);
    }
    for (; w < feature.actions.size(); ++w) {
      if (feature.actions[w].id == 0 && !is_eval_) {
        // We don't predict invalid id when training, but it should be
        // included when evaluating the model.
        continue;
      }

      int boundary = std::min(w, uniform(rng_));
      std::vector<int> pos;
      std::vector<int> neg;
      int added = 0;
      for (int b = w - 1; b >= 0; --b) {
        int id = feature.actions[b].id;
        if (id == 0) {
          // We ignore invalid id when constructing session feature.
          continue;
        }
        if (feature.actions[b].is_positive) {
          pos.push_back(id);
        }
        if (feature.actions[b].is_negative) {
          neg.push_back(id);
        }
        ++added;
        if (added >= boundary) {
          break;
        }
      }
      if (pos.empty() && !is_eval_) {
        // We ignore the example that positive session is empty when training
        continue;
      }
      float label = feature.actions[w].label;
      targets.push_back(feature.actions[w].id);
      first_videos.push_back(feature.actions[w].first_video_id);
      labels.push_back(label);
      positive_records.push_back(pos);
      negative_records.push_back(neg);

      if (label < 0.1) {
        ++num_label_level_1;
      } else if (label < 0.3) {
        ++num_label_level_2;
      } else if (label < 0.5) {
        ++num_label_level_3;
      } else if (label < 0.8) {
        ++num_label_level_4;
      } else {
        ++num_label_level_5;
      }
    }

    // Create output tensors
    Tensor* positive_records_tensor = NULL;
    Tensor* negative_records_tensor = NULL;
    Tensor* targets_tensor = NULL;
    Tensor* first_videos_tensor = NULL;
    Tensor* labels_tensor = NULL;

    int batch_size = positive_records.size();
    TensorShape positive_records_shape;
    positive_records_shape.AddDim(batch_size);
    positive_records_shape.AddDim(ws_);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, positive_records_shape,
                                             &positive_records_tensor));

    TensorShape negative_records_shape;
    negative_records_shape.AddDim(batch_size);
    negative_records_shape.AddDim(ws_);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, negative_records_shape,
                                             &negative_records_tensor));

    TensorShape targets_shape;
    targets_shape.AddDim(batch_size);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(2, targets_shape, &targets_tensor));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(3, targets_shape, &first_videos_tensor));

    TensorShape label_shape;
    label_shape.AddDim(batch_size);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, label_shape, &labels_tensor));

    // Fill output tensors
    auto matrix_positive_records = positive_records_tensor->matrix<int32>();
    auto matrix_negative_records = negative_records_tensor->matrix<int32>();
    auto flat_targets = targets_tensor->flat<int32>();
    auto flat_first_videos = first_videos_tensor->flat<int32>();
    auto flat_labels = labels_tensor->flat<float>();

    matrix_positive_records.setZero();
    matrix_negative_records.setZero();
    flat_targets.setZero();
    flat_first_videos.setZero();
    flat_labels.setZero();

    for (int i = 0; i < positive_records.size(); ++i) {
      for (int j = 0; j < positive_records[i].size(); ++j) {
        matrix_positive_records(i, j) = positive_records[i][j];
      }
    }

    for (int i = 0; i < negative_records.size(); ++i) {
      for (int j = 0; j < negative_records[i].size(); ++j) {
        matrix_negative_records(i, j) = negative_records[i][j];
      }
    }

    for (int i = 0; i < targets.size(); ++i) {
      flat_targets(i) = targets[i];
    }
    for (int i = 0; i < first_videos.size(); ++i) {
      flat_first_videos(i) = first_videos[i];
    }
    for (int i = 0; i < labels.size(); ++i) {
      flat_labels(i) = labels[i];
    }
  }

  void CalcStatistic() {
    ++global_lines_;
    auto g = global_lines_.load(std::memory_order_relaxed);
    auto level_1 = num_label_level_1.load(std::memory_order_relaxed);
    auto level_2 = num_label_level_2.load(std::memory_order_relaxed);
    auto level_3 = num_label_level_3.load(std::memory_order_relaxed);
    auto level_4 = num_label_level_4.load(std::memory_order_relaxed);
    auto level_5 = num_label_level_5.load(std::memory_order_relaxed);
    auto total = level_1 + level_2 + level_3 + level_4 + level_5;
    if (g % log_per_lines_ == 0) {
      LOG(ERROR) << "global lines = " << g;
      LOG(ERROR) << "total samples = " << total;
      LOG(ERROR) << "samples/lines = " << 1.0 * total / g;
      LOG(ERROR) << "[<10%] num_label_level_1 = " << level_1 << " ("
                 << 1.0 * level_1 / total << ")";
      LOG(ERROR) << "[<30%] num_label_level_2 = " << level_2 << " ("
                 << 1.0 * level_2 / total << ")";
      LOG(ERROR) << "[<50%] num_label_level_3 = " << level_3 << " ("
                 << 1.0 * level_3 / total << ")";
      LOG(ERROR) << "[<80%] num_label_level_4 = " << level_4 << " ("
                 << 1.0 * level_4 / total << ")";
      LOG(ERROR) << "[<=100%] num_label_level_5 = " << level_5 << " ("
                 << 1.0 * level_5 / total << ")";
    }
  }

 private:
  std::atomic<int64> global_lines_;
  std::atomic<int64> num_label_level_1;
  std::atomic<int64> num_label_level_2;
  std::atomic<int64> num_label_level_3;
  std::atomic<int64> num_label_level_4;
  std::atomic<int64> num_label_level_5;

  fe::FeatureManager feature_manager_;
  std::minstd_rand rng_;
  int ws_;
  int num_evaluate_target_per_line_;
  int log_per_lines_;
  bool is_eval_;
};

REGISTER_KERNEL_BUILDER(Name("KrankInput").Device(DEVICE_CPU), KrankInputOp);

}  // namespace tensorflow
