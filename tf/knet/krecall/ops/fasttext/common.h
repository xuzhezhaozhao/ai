#include "args.h"
#include "defines.h"
#include "dictionary.h"

void PreProcessTrainData(const std::string& train_data_path,
                         std::shared_ptr<::fasttext::Dictionary> dict);
void SaveDictionary(const std::string& dict_dir,
                    std::shared_ptr<::fasttext::Dictionary> dict);
