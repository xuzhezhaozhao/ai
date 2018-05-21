
killall -9 tensorflow_model_server
sleep 2
tensorflow_model_server --port=9000 --model_base_path=$(pwd)/model_dir > serving.log 2>&1 &
