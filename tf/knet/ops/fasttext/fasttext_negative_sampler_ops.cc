#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("FasttextNegativeSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .SetIsStateful()
    .Attr("num_true: int64")
    .Attr("num_sampled: int64")
    .Attr("unique: bool")
    .Attr("range_max: int64")
    .Attr("unigrams: tensor")
    .Attr("seed: int64 = 0")
    .Doc(R"doc(
FasttextNegativeSampler operator.
)doc");


}  // namespace tensorflow
