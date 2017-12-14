#g++ -std=c++11 -static -I. client.cc tensorflow/core/example/*.cc tensorflow/core/framework/*.cc tensorflow/core/protobuf/*.cc -L/usr/local/lib -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed -ldl -lprotobuf -lgrpc -lgrpc++ -lpthread -pthread tensorflow/core/lib/core/*.cc tensorflow_serving/apis/*.cc -o client

srcfiles=`find . -name '*.cc'`
echo ${srcfiles}

g++ -std=c++11 -static -I. -o client ${srcfiles} -lgrpc++_reflection -lgrpc++ -lgrpc  -lprotobuf -pthread
