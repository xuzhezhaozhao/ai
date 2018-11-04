
import sys

if len(sys.argv) != 3:
    raise ValueError("Usage: <input> <output>")

filename = sys.argv[1]
output = sys.argv[2]

results = []
for line in open(filename, 'r'):
    line = line.strip()
    tokens = line.split(' ')
    score = float(tokens[-1])
    id = int(tokens[-2].split('/')[-1].split('.')[0])
    results.append((id, score))

results.sort()
with open(output, 'w') as f:
    f.write('id,label\n')
    for id, score in results:
        f.write(str(id) + ',' + str(score) + '\n')
