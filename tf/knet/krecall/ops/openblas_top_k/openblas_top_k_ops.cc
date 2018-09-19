#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("OpenblasTopK")
    .Input("input: float")
    .Input("k: int32")
    .Output("values: float")
    .Output("indices: int32")
    .SetIsStateful()
    .Attr("weights: tensor")
    .Attr("biases: tensor")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
