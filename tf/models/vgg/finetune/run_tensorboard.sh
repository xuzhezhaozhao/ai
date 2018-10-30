
killall -9 tensorboard
sleep 2
tensorboard --logdir model_dir --port 8080 > tensorboard.log 2>&1 &
