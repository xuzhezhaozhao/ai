#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("KrankPredictInput")
    .Input("watched_rowkeys: string")
    .Input("rinfo1: float32")
    .Input("rinfo2: float32")
    .Input("target_rowkeys: string")
    .Input("is_video: bool")
    .Output("positive_records: int32")
    .Output("negative_records: int32")
    .Output("targets: int32")
    .SetIsStateful()
    .Attr("feature_manager_path: string = ''")
    .Attr("ws: int = 5")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
