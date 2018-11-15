from PIL import Image
import os
import skimage.io

dirname = './train'
filenames = os.listdir(dirname)

for index, filename in enumerate(filenames):
    if index % 500 == 0:
        print("rename {} ...".format(index))

    filename = os.path.join(dirname, filename)

    try:
        img = Image.open(filename)
        img2 = skimage.io.imread(filename)
    except Exception as e:
        print("catch except: {}".format(e))
        print("delete {} ...".format(filename))
        os.remove(filename)
        continue

    if (len(img2.shape) != 3) or (img2.shape[2] != 3):
        print("shape error, filename {}".format(filename))
        continue

    img_type = img.format.lower()
    img_mode = img.mode.lower()
    allowed_types = set(['jpeg', 'jpg', 'png', 'bmp'])
    allowed_modes = set(['rgb'])
    if img_type not in allowed_types or img_mode not in allowed_modes:
        print("type/mode error: {}, {}, {}"
              .format(filename, img_type, img_mode))
        print("delete {} ...".format(filename))
        os.remove(filename)
        continue

    if len(img.size) != 2:
        print("shape error: " + filename + ", " + str(img.size))
        print("delete {} ...".format(filename))
        os.remove(filename)
        continue

    os.rename(filename, filename + '.' + img_type)
