from PIL import Image
import os
import threading

output_dir = 'train'
os.system('mkdir -p ' + output_dir)

nthreads = 64


def wget_thread_body(tid, urls, rowkeys, start_idx):
    filenames = []
    for idx, url in enumerate(urls):
        if idx % 100 == 0:
            print("tid {}, wget {} ...".format(tid, idx+1))

        filename = os.path.join(output_dir, rowkeys[idx] + '.'
                                + str(start_idx + idx))
        cmd = 'wget ' + "'" + url + "'" + ' -O ' + filename + '> wget.' + str(tid) + '.log 2>&1'
        try:
            os.system(cmd)
            filenames.append(filename)
        except Exception as e:
            print("catch Exception: {}".format(e))


rowkeys = []
urls = []
for index, line in enumerate(open('./cover.txt')):
    if index < 1:
        continue
    line = line.strip()
    tokens = line.split('\t')
    rowkeys.append(tokens[0])
    urls.append(tokens[1])

chunk_size = len(rowkeys) / nthreads
workers = []
print("total rowkeys = {}".format(len(rowkeys)))
print("chunk_size = {}".format(chunk_size))
for tid in range(nthreads):
    sub_rowkeys = rowkeys[tid*chunk_size:(tid+1)*chunk_size]
    sub_urls = urls[tid*chunk_size:(tid+1)*chunk_size]

    worker = threading.Thread(
        target=wget_thread_body,
        args=(tid, sub_urls, sub_rowkeys, tid*chunk_size))
    worker.start()
    workers.append(worker)

for worker in workers:
    worker.join()
