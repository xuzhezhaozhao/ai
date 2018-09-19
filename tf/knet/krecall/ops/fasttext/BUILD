
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "fasttext_example_generate_ops.so",
    srcs = ["fasttext_example_generate_kernels.cc",
            "fasttext_example_generate_ops.cc",
            "dictionary.cc"],
    deps = ["//tensorflow/core/user_ops:headers"],
)
cc_library(
    name = "headers",
    hdrs = glob(["*.h"]),
)
