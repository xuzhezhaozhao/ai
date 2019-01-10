
set -e

srcfiles=`find . -name '*.cc'`
echo ${srcfiles}

g++ -std=c++11 -static -I. -o client ${srcfiles} \
    -lgrpc++ -lgrpc -lprotobuf -pthread -ldl -lz
