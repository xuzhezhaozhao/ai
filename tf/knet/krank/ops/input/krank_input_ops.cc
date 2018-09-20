#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("KrankInput")
    .Input("input: string")
    .Output("positive_records: int32")
    .Output("negative_records: int32")
    .Output("targets: int32")
    .Output("labels: int32")
    .SetIsStateful()
    .Attr("feature_manager_path: string = ''")
    .Attr("ws: int = 5")
    .Attr("is_eval: bool = false")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
