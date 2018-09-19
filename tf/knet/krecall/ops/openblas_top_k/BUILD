
package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

cc_library(
    name = "openblas_top_k_ops",
    visibility = ["//visibility:public"],
    srcs = glob(["*.h", "*.cc", "deps/openblas/include/*.h"]),
    copts = ["-std=c++11"],
    linkopts = ["-L/home/xzz/tf_serving/tensorflow_serving/openblas_top_k/deps/openblas/lib/", "-lopenblas"], # -L should be absolutely path
    deps = ["@org_tensorflow//tensorflow/core:framework_headers_lib",
            "@org_tensorflow//tensorflow/core/util/ctc",
            "@org_tensorflow//third_party/eigen3",
    ],
    alwayslink = 1,
)
