
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("KrankInput")
    .Input("input: string")
    .Output("positive_records: int32")
    .Output("negative_records: int32")
    .Output("targets: int64")
    .Output("labels: int64")
    .SetIsStateful()
    .Attr("feature_manager_path: string = ''")
    .Attr("ws: int = 5")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
