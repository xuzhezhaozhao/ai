from PIL import Image
import os

dirname = './train'
filenames = os.listdir(dirname)

for index, filename in enumerate(filenames):
    if index % 500 == 0:
        print("rename {} ...".format(index))

    filename = os.path.join(dirname, filename)

    try:
        img = Image.open(filename)
    except Exception as e:
        print("catch except: {}".format(e))
        print("delete {} ...".format(filename))
        os.remove(filename)
        continue

    img_type = img.format.lower()
    allowed_types = set(['jpeg', 'jpg', 'png', 'bmp'])
    if img_type not in allowed_types:
        print("type error: " + filename + ", " + str(img_type))

    if len(img.size) != 2:
        print("shape error: " + filename + ", " + str(img.size))

    os.rename(filename, filename + '.' + img_type)
