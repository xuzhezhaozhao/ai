
import os

dirname = './leg_images'
output_dirname = 'output'
filenames = os.listdir(dirname)

for index, filename in enumerate(filenames):
    if not filename.endswith('.jpg') and not filename.endswith('.jpeg'):
        print(filename + ' is not jpg or jpeg.')
        continue
    newname = 'leg_' + str(index) + '.jpg'
    os.rename(os.path.join(dirname, filename),
              os.path.join(output_dirname, newname))
