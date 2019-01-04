
with open('./rowkey.csv', 'w') as f:
    for line in open('./kd_video_comments.csv'):
        i = line.find('\t')
        rowkey = line[:i]
        f.write(rowkey)
        f.write('\n')
