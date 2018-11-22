
cd protos
bash gen.sh

cd python_out
rm -rf client.py
cp ../../client_rank.py .
python client_rank.py
