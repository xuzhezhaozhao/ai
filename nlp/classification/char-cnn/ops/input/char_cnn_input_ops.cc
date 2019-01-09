
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("CharCNNInput")
    .Input("input: string")
    .Output("char_ids: int32")
    .Output("label: int64")
    .SetIsStateful()
    .Attr("char_dict: tensor")
    .Attr("label_dict: tensor")
    .Attr("label_str: string = '__label__'")
    .Attr("max_length: int = 32")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
