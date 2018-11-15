from PIL import Image
import os

output_dir = 'train'
print("wget ...")
for index, line in enumerate(open('./cover.txt')):
    if index < 1:
        continue

    if index % 1000 == 0:
        print("wget {} ...".format(index))

    line = line.strip()
    tokens = line.split('\t')
    rowkey = tokens[0]
    url = tokens[1]
    filename = os.path.join(output_dir, rowkey + '_' + str(index))
    cmd = 'wget ' + "'" + url + "'" + ' -O ' + filename + ' > wget.log 2>&1'
    try:
        os.system('mkdir -p ' + output_dir)
        os.system(cmd)
    except Exception as e:
        print("catch Exception: {}".format(e))

print("wget done.")

print("rename ...")
filenames = os.listdir(output_dir)
for index, name in enumerate(filenames):
    if index % 1000 == 0:
        print("rename {} ...".format(index))

    img = Image.open(filename)
    img_type = img.format.lower()
    allowed_types = set(['jpeg', 'jpg', 'png', 'bmp'])
    if img_type not in allowed_types:
        print("type error: " + filename + ", " + str(img_type))

    if len(img.size) != 2:
        print("shape error: " + filename + ", " + str(img.size))

    os.rename(filename, filename + '.' + img_type)
print("rename done")
