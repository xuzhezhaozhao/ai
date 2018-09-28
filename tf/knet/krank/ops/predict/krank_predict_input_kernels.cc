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

#include "fe/feature_manager.h"

namespace tensorflow {
class KrankPredictInputOp : public OpKernel {
 public:
  explicit KrankPredictInputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), feature_manager_(), ws_(0) {
    LOG(INFO) << "Init KrankPredictInputOp ...";

    std::string fm;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_manager", &fm));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ws", &ws_));
    feature_manager_.load_from_string(fm);

    LOG(INFO) << "Init KrankPredictInputOp OK";
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& watched_rowkeys_tensor = ctx->input(0);
    const Tensor& rinfo1_tensor = ctx->input(1);
    const Tensor& rinfo2_tensor = ctx->input(2);
    const Tensor& target_rowkeys_tensor = ctx->input(3);
    const Tensor& is_video_tensor = ctx->input(4);
    const TensorShape& watched_rowkeys_shape = watched_rowkeys_tensor.shape();
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(watched_rowkeys_shape),
                errors::InvalidArgument("input expects a Matrix."));

    int batch_size = watched_rowkeys_shape.dim_size(0);
    int watched_size = watched_rowkeys_shape.dim_size(1);
    int targets_size = target_rowkeys_tensor.shape().dim_size(1);

    OP_REQUIRES(ctx, rinfo1_tensor.shape() == watched_rowkeys_shape,
                errors::InvalidArgument("rinfo1 shape not matched."));
    OP_REQUIRES(ctx, rinfo2_tensor.shape() == watched_rowkeys_shape,
                errors::InvalidArgument("rinfo2 shape not matched."));
    OP_REQUIRES(ctx, is_video_tensor.shape() == watched_rowkeys_shape,
                errors::InvalidArgument("is_video shape not matched."));

    auto matrix_watched_rowkeys = watched_rowkeys_tensor.matrix<std::string>();
    auto matrix_rinfo1 = rinfo1_tensor.matrix<float>();
    auto matrix_rinfo2 = rinfo2_tensor.matrix<float>();
    auto matrix_target_rowkeys = target_rowkeys_tensor.matrix<std::string>();
    auto is_video_matrix = is_video_tensor.matrix<int64>();

    Tensor* positive_records_tensor = NULL;
    Tensor* negative_records_tensor = NULL;
    Tensor* targets_tensor = NULL;
    Tensor* is_target_in_dict_tensor = NULL;
    Tensor* num_positive_tensor = NULL;
    Tensor* num_negative_tensor = NULL;
    TensorShape positive_records_shape;
    positive_records_shape.AddDim(batch_size * targets_size);
    positive_records_shape.AddDim(ws_);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, positive_records_shape,
                                             &positive_records_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, positive_records_shape,
                                             &negative_records_tensor));

    TensorShape targets_shape;
    targets_shape.AddDim(batch_size * targets_size);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(2, targets_shape, &targets_tensor));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(3, targets_shape, &is_target_in_dict_tensor));

    TensorShape num_shape;
    num_shape.AddDim(batch_size);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, num_shape, &num_positive_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(5, num_shape, &num_negative_tensor));

    auto matrix_positive_records = positive_records_tensor->matrix<int32>();
    auto matrix_negative_records = negative_records_tensor->matrix<int32>();
    auto flat_targets = targets_tensor->flat<int32>();
    auto flat_is_target_in_dict = is_target_in_dict_tensor->flat<int32>();
    auto flat_num_positive = num_positive_tensor->flat<int32>();
    auto flat_num_negative = num_negative_tensor->flat<int32>();

    matrix_positive_records.setZero();
    matrix_negative_records.setZero();
    flat_targets.setZero();
    flat_is_target_in_dict.setZero();
    flat_num_positive.setZero();
    flat_num_negative.setZero();

    for (int b = 0; b < batch_size; ++b) {
      fe::RawFeature raw_feature;
      for (int i = 0; i < watched_size; ++i) {
        if (matrix_watched_rowkeys(b, i) == "") {
          continue;
        }

        fe::UserAction action(matrix_rinfo1(b, i), matrix_rinfo2(b, i),
                              matrix_watched_rowkeys(b, i),
                              is_video_matrix(b, i));
        raw_feature.actions.push_back(action);
      }
      fe::TransformedFeature feature = feature_manager_.transform(raw_feature);

      // fill targets
      for (int i = 0; i < targets_size; ++i) {
        int index = b * targets_size + i;
        int id = feature_manager_.getRowkeyId(matrix_target_rowkeys(b, i));
        flat_targets(index) = id;
        flat_is_target_in_dict(index) = (id == 0 ? 0 : 1);

        int added = 0;
        for (int w = 0; w < feature.actions.size(); ++w) {
          if (added >= ws_) {
            break;
          }
          const auto& action = feature.actions[w];
          if (action.id == 0) {
            continue;
          }
          if (action.is_positive) {
            matrix_positive_records(index, added) = action.id;
            ++added;
          }
        }
        flat_num_positive(b) = added;

        added = 0;
        for (int w = 0; w < feature.actions.size(); ++w) {
          if (added >= ws_) {
            break;
          }
          const auto& action = feature.actions[w];
          if (action.id == 0) {
            continue;
          }
          if (action.is_negative) {
            matrix_negative_records(index, added) = action.id;
            ++added;
          }
        }
        flat_num_negative(b) = added;
      }
    }
  }

 private:
  fe::FeatureManager feature_manager_;
  int ws_;
};

REGISTER_KERNEL_BUILDER(Name("KrankPredictInput").Device(DEVICE_CPU),
                        KrankPredictInputOp);
}  // namespace tensorflow
