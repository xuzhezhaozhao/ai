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

#include <fstream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "matrix.h"
#include "openblas_util.h"
#include "vector.h"

namespace tensorflow {
class OpenblasTopKOp : public OpKernel {
 public:
  explicit OpenblasTopKOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    LOG(ERROR) << "Init OpenblasTopKOp ...";
    Tensor weights_tensor;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("weights", &weights_tensor));
    LoadWeights(ctx, weights_tensor);

    Tensor biases_tensor;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("biases", &biases_tensor));
    LoadBiases(ctx, biases_tensor);

    OP_REQUIRES(
        ctx, weights_.rows() == biases_.size(),
        errors::InvalidArgument("weights and biases dimension not matched."));

    LOG(ERROR) << "Init OpenblasTopKOp OK";
  }

  void Compute(OpKernelContext* ctx) override {
    // calculate matmul using openblas
    const Tensor& input_tensor = ctx->input(0);
    const TensorShape& input_shape = input_tensor.shape();
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(input_shape),
                errors::InvalidArgument("input expects a [batch, dim] Matrix."));
    OP_REQUIRES(
        ctx, input_shape.dim_size(0) == 1,
        errors::InvalidArgument("Expect Input tensor's dim 0 be 1, but is ",
                                input_shape.dim_size(0)));
    OP_REQUIRES(ctx, input_shape.dim_size(1) == weights_.cols(),
                errors::InvalidArgument("Expect Input tensor's dim 1 be ",
                                        weights_.cols(), ", but is ",
                                        input_shape.dim_size(1)));
    const Tensor& k_tensor = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(k_tensor.shape()),
                errors::InvalidArgument("Arg k expects a scalar"));
    auto flat_k = k_tensor.flat<int32>();
    int32 k = flat_k(0);

    const float* vec = input_tensor.flat<float>().data();
    int sz = input_tensor.flat<float>().size();
    int n_classes = weights_.rows();
    fasttext::Vector logits(n_classes);
    cblas_vec_dot_matrix(vec, sz, weights_, logits);
    logits.addVector(biases_);

    // calculate top k
    TensorShape output_shape;
    output_shape.AddDim(k);
    Tensor* values_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &values_tensor));
    Tensor* indices_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, output_shape, &indices_tensor));
    auto values = values_tensor->flat<float>();
    auto indices = indices_tensor->flat<int32>();
    std::vector<std::pair<float, int>> heap(n_classes);
    for (int i = 0; i < n_classes; ++i) {
      heap[i].first = logits[i];
      heap[i].second = i;
    }
    std::make_heap(heap.begin(), heap.end());
    // TODO ban set
    std::unordered_set<int> ban_set;
    int32_t i = 0;
    size_t poped = 0;
    while (i < k && heap.size() > 0) {
      auto& top = heap.front();
      auto it = ban_set.find(top.second);
      if (it == ban_set.end()) {
        values(i) = top.first;
        indices(i) = top.second;
        i++;
      }
      pop_heap(heap.begin(), heap.end() - poped);
      ++poped;
    }
  }

 private:
  void LoadWeights(OpKernelConstruction* ctx, const Tensor& weights_tensor) {
    auto& weights_shape = weights_tensor.shape();
    auto flat_weights = weights_tensor.flat<float>();
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weights_shape),
                errors::InvalidArgument("weights expects a Matrix."));
    int row = weights_shape.dim_size(0);
    int cols = weights_shape.dim_size(1);

    weights_ = fasttext::Matrix(row, cols);
    for (int i = 0; i < flat_weights.size(); ++i) {
      weights_.data_[i] = flat_weights(i);
    }

    LOG(ERROR) << "Load weights shape: " << weights_.rows() << ", "
               << weights_.cols();
    for (int i = 0; i < weights_.size() && i < 50; ++i) {
      LOG(ERROR) << weights_.data_[i];
    }

    weights_.convertColMajor();
  }

  void LoadBiases(OpKernelConstruction* ctx, const Tensor& biases_tensor) {
    auto& biases_shape = biases_tensor.shape();
    auto flat_biases = biases_tensor.flat<float>();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(biases_shape),
                errors::InvalidArgument("biases expects a Vector."));
    int sz = flat_biases.size();
    biases_ = fasttext::Vector(sz);
    for (int i = 0; i < sz; ++i) {
      biases_.data_[i] = flat_biases(i);
    }

    LOG(ERROR) << "Load biases shape: " << biases_.size();
    for (int i = 0; i < biases_.size() && i < 20; ++i) {
      LOG(ERROR) << biases_.data_[i];
    }
  }

  fasttext::Matrix weights_;
  fasttext::Vector biases_;
};

REGISTER_KERNEL_BUILDER(Name("OpenblasTopK").Device(DEVICE_CPU),
                        OpenblasTopKOp);

}  // namespace tensorflow
