
add_library(
  libfasttext
  args.cc
  dictionary.cc
  fasttext.cc
  fasttext_api.cc
  matrix.cc
  mockmain.cc
  model.cc
  productquantizer.cc
  qmatrix.cc
  utils.cc
  vector.cc
)

add_executable(fasttext main.cc)
target_link_libraries(fasttext libfasttext)
