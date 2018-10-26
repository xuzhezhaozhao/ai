import os
import skimage.io
import tensorflow as tf

dirnames = ['./train', './test']
error_dirname = './error'

for dirname in dirnames:
    print("check " + dirname)
    filenames = os.listdir(dirname)
    for index, name in enumerate(filenames):
        filename = os.path.join(dirname, name)
        try:
            img = skimage.io.imread(filename)
            if len(img.shape) != 3 or img.shape[2] != 3:
                print("shape error: " + filename + ", shape " + str(img.shape))
                os.rename(filename, os.path.join(error_dirname, name))
            elif img.shape[0] < 224 or img.shape[1] < 224:
                print("too little: " + filename + ", shape " + str(img.shape))
                os.rename(filename, os.path.join(error_dirname, name))
        except Exception:
            print("decode error: " + filename)
            os.rename(filename, os.path.join(error_dirname, name))


print("check with tf ...")
sess = tf.Session()
for dirname in dirnames:
    print("check " + dirname)
    filenames = os.listdir(dirname)
    for index, name in enumerate(filenames):
        filename = os.path.join(dirname, name)

        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_image(img_string, channels=3)

        img = sess.run(img_decoded)
        if len(img.shape) != 3 or img.shape[2] != 3:
            print("shape error: " + filename + ", shape " + str(img.shape))
            os.rename(filename, os.path.join(error_dirname, name))
        elif img.shape[0] < 224 or img.shape[1] < 224:
            print("too little: " + filename + ", shape " + str(img.shape))
            os.rename(filename, os.path.join(error_dirname, name))

        if index % 100 == 0:
            print("process " + str(index))
