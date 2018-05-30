#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("OpenblasTopK")
    .Input("input: float")
    .Input("k: int32")
    .Output("values: float")
    .Output("indices: int32")
    .SetIsStateful()
    .Attr("weights_path: string")
    .Attr("biases_path: string")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
