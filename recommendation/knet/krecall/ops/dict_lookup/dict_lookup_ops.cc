#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("DictLookup")
    .Input("input: string")
    .Output("ids: int64")
    .Output("num_in_dict: int64")
    .SetIsStateful()
    .Attr("dict: tensor")
    .Attr("output_ws: int")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
