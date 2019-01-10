# compile in tensorflow/core/user_ops

bazel build --config opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" //tensorflow/core/user_ops:fasttext_example_generate_ops.so
