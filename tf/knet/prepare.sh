set -e
cd ./ops/fasttext/
./compile.sh

cd ../..
cd ./ops/dict_lookup/
./compile.sh

cd ../../
cd ./ops/openblas_top_k
./compile.sh
