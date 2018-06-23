#include <fstream>

#include <gflags/gflags.h>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

#include "args.h"
#include "defines.h"
#include "dictionary.h"

DEFINE_string(tfrecord_file, "", "");
DEFINE_int32(ws, 5, "");
DEFINE_int32(min_count, 5, "");
DEFINE_double(t, 0.0001, "");
DEFINE_int32(ntargets, 1, "");
DEFINE_double(sample_dropout, 0.5, "");
DEFINE_string(saved_dict_file, "", "");
DEFINE_string(input_file, "", "");

class WritableFile;

void load_dictionary(const std::string &saved_dict_file,
                     std::shared_ptr<::fasttext::Dictionary> dict) {
  LOG(INFO) << "Load dictionary from " << saved_dict_file << " ...";
  std::ifstream ifs(saved_dict_file);
  if (!ifs.is_open()) {
    LOG(FATAL) << "Error: Open " << saved_dict_file << " failed." << std::endl;
  }
  dict->load(ifs);
  if (ifs.fail()) {
    LOG(FATAL) << "Error: Load dict failed." << std::endl;
  }
  ifs.close();
  LOG(INFO) << "Load dictionary OK.";
}

inline int transform_id(int id) { return id + 1; }

void fill_example(const std::vector<int> &inst,
                  std::shared_ptr<::fasttext::Args> args,
                  ::tensorflow::Example *example) {
  auto features = example->mutable_features();
  auto feature = features->mutable_feature();
  ::tensorflow::Feature records;
  ::tensorflow::Feature label;
  for (int t = 0; t < args->ntargets; ++t) {
    int index = inst.size() - args->ntargets + t;
    label.mutable_int64_list()->add_value(transform_id(inst[index]));
  }
  (*feature)["label"] = label;

  // fill records
  auto records_int_list = records.mutable_int64_list();
  for (int i = 0; i < inst.size() - args->ntargets; ++i) {
    records_int_list->add_value(transform_id(inst[i]));
  }

  // padding records
  for (int i = inst.size() - args->ntargets; i < args->ws; ++i) {
    records_int_list->add_value(0);
  }
  (*feature)["records"] = records;
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  ::tensorflow::Env *env = ::tensorflow::Env::Default();
  std::unique_ptr<::tensorflow::WritableFile> file;
  TF_CHECK_OK(env->NewWritableFile(FLAGS_tfrecord_file, &file));
  ::tensorflow::io::RecordWriter writer(file.get());

  std::shared_ptr<::fasttext::Args> args = std::make_shared<::fasttext::Args>();
  args->ws = FLAGS_ws;
  args->min_count = FLAGS_min_count;
  args->t = FLAGS_t;
  args->ntargets = FLAGS_ntargets;
  args->sample_dropout = FLAGS_sample_dropout;

  std::shared_ptr<::fasttext::Dictionary> dict =
      std::make_shared<::fasttext::Dictionary>(args);
  load_dictionary(FLAGS_saved_dict_file, dict);

  int64_t line_processed = 0;
  int64_t total = 0;

  std::ifstream ifs(FLAGS_input_file);
  if (!ifs.is_open()) {
    LOG(FATAL) << "open " << FLAGS_input_file << " failed.";
  }
  std::string line;
  std::vector<int32_t> words;
  std::vector<int> inst;
  std::minstd_rand rng(time(NULL));
  while (!ifs.eof()) {
    ++line_processed;
    if (line_processed % 10000) {
      LOG(INFO) << line_processed << " processed," << total << " examples.";
    }

    std::getline(ifs, line);
    if (line == "") {
      continue;
    }
    words.clear();
    std::stringstream ss(line);
    int ntokens = dict->getLine(ss, words, rng);
    std::uniform_int_distribution<> uniform(1, args->ws);
    std::uniform_real_distribution<> dropout_uniform(0, 1);

    // genearte examples
    for (int w = 1; w < words.size(); w++) {
      inst.clear();
      if (dropout_uniform(rng) < args->sample_dropout) {
        continue;
      }
      // use words[w] as the first label
      int32_t boundary = std::min(w, uniform(rng));
      for (int c = -boundary; c < 0; c++) {
        inst.push_back(words[w + c]);
      }
      inst.push_back(words[w]);  // add label

      // TODO random select ntargets-1 labels
      for (int i = 0; i < args->ntargets - 1; ++i) {
        int t = w + 1 + i;
        if (t >= words.size()) {
          t = w;
        }
        inst.push_back(words[t]);
      }
      // Create Example and write to tfrecord file
      ::tensorflow::Example example;
      fill_example(inst, args, &example);
      std::string serialized;
      example.SerializeToString(&serialized);
      TF_CHECK_OK(writer.WriteRecord(serialized));
      ++total;
    }
  }

  // flush
  TF_CHECK_OK(writer.Flush());
  TF_CHECK_OK(writer.Close());

  return 0;
}
