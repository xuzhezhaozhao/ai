import os

basedir = os.path.split(os.path.realpath(__file__))[0]
test_dir = os.path.join(basedir, 'test/')

with open('test.txt', 'w') as f:
    for filename in sorted(os.listdir(test_dir)):
        fullname = os.path.join(test_dir, filename)
        f.write(fullname + '\n')
