#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("FasttextNegativeSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .SetIsStateful()
    .Attr("num_true: int")
    .Attr("num_sampled: int")
    .Attr("unique: bool")
    .Attr("range_max: int")
    .Attr("num_reserved_ids: int = 0")
    .Attr("unigrams: tensor")
    .Attr("seed: int = 0")
    .Doc(R"doc(
FasttextNegativeSampler operator.
  true_classes: Tensor of shape [batch, num_true].
  sampled_candidates: Tensor of shape [batch, num_sampled].
)doc");


}  // namespace tensorflow
