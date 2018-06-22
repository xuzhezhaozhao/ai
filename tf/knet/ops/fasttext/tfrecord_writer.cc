
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

class WritableFile;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: <tfrecord_file>" << std::endl;
    exit(-1);
  }
  std::string fname = argv[1];

  ::tensorflow::Env* env = ::tensorflow::Env::Default();
  std::unique_ptr<::tensorflow::WritableFile> file;
  TF_CHECK_OK(env->NewWritableFile(fname, &file));
  ::tensorflow::io::RecordWriter writer(file.get());

  // Create Example and write to tfrecord file
  ::tensorflow::Example example;
  auto features = example.mutable_features();
  auto feature = features->mutable_feature();
  ::tensorflow::Feature words;
  auto bytes_list = words.mutable_bytes_list();
  int ws = 20;
  for (int i = 0; i < ws; ++i) {
    bytes_list->add_value("a");
  }
  for (int i = bytes_list->value_size(); i < ws; ++i) {
    bytes_list->add_value("");
  }
  (*feature)["words"] = words;

  std::string serialized;
  example.SerializeToString(&serialized);
  TF_CHECK_OK(writer.WriteRecord(serialized));

  // flush
  TF_CHECK_OK(writer.Flush());

  return 0;
}
