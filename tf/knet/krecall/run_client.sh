
cd protos
bash gen.sh

cd python_out
rm -rf client.py
cp ../../client.py .
python client.py
