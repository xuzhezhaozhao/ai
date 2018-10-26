import os

output_dir = 'ugc_images'
for index, line in enumerate(open('./ugc_images.txt')):
    line = line.strip()
    if index >= 3000:
        break
    name = output_dir + '/' + 'ugc_' + str(index) + '.jpg'
    cmd = 'wget ' + line + ' -O ' + name
    print(cmd)
    os.system('mkdir -p ' + output_dir)
    os.system(cmd)
