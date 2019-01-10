
import sys

predict_output = './predict_output.txt'

if len(sys.argv) == 2:
    predict_output = sys.argv[1]

for line in open(predict_output, 'r'):
    line = line.strip()
    if line == '':
        continue
    tokens = line.split(' ')
    score = float(tokens[2])
    t = tokens[0].split('/')[-1].split('.')[0]

    if t == 'dog':
        if score < 0.5:
            print(line)
    else:
        if score > 0.5:
            print(line)
