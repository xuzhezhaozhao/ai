
killall -9 tensorboard
sleep 2
tensorboard --logdir model_dir > tensorboard.log 2>&1 &
