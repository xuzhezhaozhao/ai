import skimage.io

cnt = 0
for filename in open('./train.txt'):
    filename = filename.strip()
    try:
        img = skimage.io.imread(filename)
        if len(img.shape) != 3 or img.shape[2] != 3:
            print("shape error: " + filename + ", shape " + str(img.shape))
    except Exception:
        print("decode error: " + filename)
    cnt += 1
    if cnt % 2000 == 0:
        print("{} processed ...".format(cnt))
