
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("TextCNNInput")
    .Input("input: string")
    .Output("word_ids: int32")
    .Output("label: int64")
    .SetIsStateful()
    .Attr("word_dict: tensor")
    .Attr("label_dict: tensor")
    .Attr("label_str: string = '__label__'")
    .Attr("max_length: int = 32")
    .Attr("log_per_lines: int = 10000")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
