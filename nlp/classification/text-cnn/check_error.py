
from __future__ import print_function

import sys


if len(sys.argv) != 3:
    print("Usage: <predict> <target>")
    sys.exit(-1)

for line1, line2 in zip(open(sys.argv[1]), open(sys.argv[2])):
    label, score = line1.strip().split()
    target = line2.split(' ')[0]
    if label != target:
        print(score[:4] + ' ' + line2.strip())
