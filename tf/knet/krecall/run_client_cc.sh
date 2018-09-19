
cd protos
bash gen.sh

cd cpp_out
cp ../../compile.sh .
cp ../../client.cc .
bash compile.sh

./client 127.0.0.1:9000
