from PIL import Image
import os

dirnames = ['./train', './test']

for dirname in dirnames:
    print("check " + dirname)
    filenames = os.listdir(dirname)
    for index, name in enumerate(filenames):
        filename = os.path.join(dirname, name)
        img = Image.open(filename)
        t = img.format.lower()
        if t != 'jpeg' and t != 'jpg' and t != 'png' and t != 'bmp':
            print("type error: " + filename + ", " + str(t))

        if len(img.size) != 2:
            print("shape error: " + filename + ", " + str(img.size))
