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
      : OpKernel(ctx), feature_manager_(), ws_(0) {
    LOG(INFO) << "Init KrankInputOp ...";

    rng_.seed(time(NULL));

    std::string fm_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_manager_path", &fm_path));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ws", &ws_));
    feature_manager_.load(fm_path);

    LOG(INFO) << "Init KrankInputOp OK.";
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    auto flat_input = input_tensor.flat<std::string>();
    OP_REQUIRES(ctx, flat_input.size() == 1,
                errors::InvalidArgument("input size not 1"));
    const std::string& s = flat_input(0);

    fe::TransformedFeature feature = feature_manager_.transform(s);

    std::vector<std::vector<int>> positive_records;
    std::vector<std::vector<int>> negative_records;
    std::vector<int64> targets;
    std::vector<bool> labels;
    std::uniform_int_distribution<> uniform(1, ws_);
    for (int w = 1; w < feature.actions.size(); ++w) {
      if (feature.actions[w].id == 0) {
        // ignore non-valid id
        continue;
      }

      int boundary = std::min(w, uniform(rng_));
      std::vector<int> pos;
      std::vector<int> neg;
      int added = 0;
      for (int b = w - 1; b >= 0; --b) {
        int id = feature.actions[b].id;
        if (id == 0) {
          // ignore non-valid id
          continue;
        }
        if (feature.actions[b].label) {
          pos.push_back(id);
        }
        if (feature.actions[b].unlike) {
          neg.push_back(id);
        }
        ++added;

        if (added >= boundary) {
          break;
        }
      }

      targets.push_back(feature.actions[w].id);
      labels.push_back(feature.actions[w].label);
      positive_records.push_back(pos);
      negative_records.push_back(neg);
    }

    // Create output tensors
    Tensor* positive_records_tensor = NULL;
    Tensor* negative_records_tensor = NULL;
    Tensor* targets_tensor = NULL;
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

    TensorShape label_shape;
    label_shape.AddDim(batch_size);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, label_shape, &labels_tensor));

    // Fill output tensors
    auto matrix_positive_records = positive_records_tensor->matrix<int32>();
    auto matrix_negative_records = negative_records_tensor->matrix<int32>();
    auto flat_targets = targets_tensor->flat<int32>();
    auto flat_labels = labels_tensor->flat<int32>();

    matrix_positive_records.setZero();
    matrix_negative_records.setZero();
    flat_targets.setZero();
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
    for (int i = 0; i < labels.size(); ++i) {
      flat_labels(i) = (labels[i] ? 1 : 0);
    }
  }

 private:
  fe::FeatureManager feature_manager_;
  std::minstd_rand rng_;
  int ws_;
};

REGISTER_KERNEL_BUILDER(Name("KrankInput").Device(DEVICE_CPU), KrankInputOp);

}  // namespace tensorflow
