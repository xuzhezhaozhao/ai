
python ensemble_predict.py ../text-cnn/predict.txt ./predict.txt > ensemble_predict.txt
python check_error.py ./ensemble_predict.txt ../../../datasets/kd_video_comments-dataset/data/char-cnn/test.txt > ensemble_error.txt
