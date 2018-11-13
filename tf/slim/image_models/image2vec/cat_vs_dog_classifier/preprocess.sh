
OPTS=''

awk '{print $1}' train.txt > train_images.txt
awk '{print $2}' train.txt > train_labels.txt
../image2vec ${OPTS} --input train_images.txt --output train_features.txt

awk '{print $1}' validation.txt > validation_images.txt
awk '{print $2}' validation.txt > validation_labels.txt
../image2vec ${OPTS} --input validation_images.txt --output validation_features.txt

awk '{print $1}' test.txt > test_images.txt
../image2vec ${OPTS} --input test_images.txt --output test_features.txt
