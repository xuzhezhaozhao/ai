
import sys
from collections import Counter

if len(sys.argv) != 4:
    print("Usage: <neg_dict> <pos_file> <neg_file>")
    sys.exit(-1)

keywords = list(set(keyword.strip().lower() for keyword in open(sys.argv[1])))
dic = Counter()
print("keywords:")
for key in keywords:
    print("{}").format(key)

with open(sys.argv[2], 'w') as fpos, open(sys.argv[3], 'w') as fneg:
    for line in open('./comments/kd_video_comments.csv'):
        ispos = True
        for keyword in keywords:
            if line.lower().find(keyword) >= 0:
                ispos = False
                dic[keyword] += 1
                break
        if ispos:
            fpos.write(line)
        else:
            fneg.write(line)

print("keywords count\n")
for key in dic:
    print("{}: {}\n".format(key, dic[key]))
