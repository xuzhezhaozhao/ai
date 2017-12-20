# fastText

[fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification.

封装成 API 形式

用法:

cmake 添加
```
add_subdirectory(fasttext)

...
target_link_libraries(<exe> libfasttext)
```

调用 api
```
#include "fasttext_api.h"

  ...

  fasttext::FastTextApi fapi;
  fapi.LoadModel(argv[1]);
  std::string test_data = "我喜欢姚明, 想看球赛";
  auto preditions = fapi.Predict(test_data, 10);

  for (auto it = preditions.cbegin(); it != preditions.cend(); ++it) {
    std::cout << it->second << " : " << it->first << std::endl;
  }

  ...

```

注意: 由于依赖 jieba 分词，分词的词典文件夹 dict/ 必须放在当前工作路径的父目录上
