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
    std::string weights_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("weights_path", &weights_path));

    std::string biases_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("biases_path", &biases_path));

    std::ifstream weights_in(weights_path);
    OP_REQUIRES(
        ctx, weights_in.is_open(),
        errors::Unavailable("'" + weights_path + "'" + " open failed."));
    weights_.load(weights_in);
    weights_.convertColMajor();

    std::ifstream biases_in(biases_path);
    OP_REQUIRES(ctx, biases_in.is_open(),
                errors::Unavailable("'" + biases_path + "'" + " open failed."));
    biases_.load(biases_in);
  }

  void Compute(OpKernelContext* ctx) override {
    // calculate matmul using openblas
    // TODO check dimention
    const Tensor& input_tensor = ctx->input(0);
    const Tensor& k_tensor = ctx->input(1);
    int32 k = k_tensor.flat<int32>()(0);

    const TensorShape& input_shape = input_tensor.shape();
    const float* vec = input_tensor.flat<float>().data();
    int sz = input_tensor.flat<float>().size();
    int n_classes = weights_.m_;
    fasttext::Vector logits(n_classes);
    cblas_vec_dot_matrix(vec, sz, weights_, logits);
    logits.addVector(biases_);

    // calculate top k
    TensorShape output_shape;
    output_shape.AddDim(n_classes);
    Tensor* values_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &values_tensor));
    Tensor* indices_tensor = NULL;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, output_shape, &indices_tensor));
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
  fasttext::Matrix weights_;
  fasttext::Vector biases_;
};

REGISTER_KERNEL_BUILDER(Name("OpenblasTopK").Device(DEVICE_CPU),
                        OpenblasTopKOp);

}  // namespace tensorflow
