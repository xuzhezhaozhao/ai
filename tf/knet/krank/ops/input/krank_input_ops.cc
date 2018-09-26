#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("KrankInput")
    .Input("input: string")
    .Output("positive_records: int32")
    .Output("negative_records: int32")
    .Output("targets: int32")
    .Output("labels: float")
    .SetIsStateful()
    .Attr("feature_manager_path: string = ''")
    .Attr("ws: int = 5")
    .Attr("num_evaluate_target_per_line: int = 1")
    .Attr("log_per_lines: int = 10000")
    .Attr("is_eval: bool = false")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
