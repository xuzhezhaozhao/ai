
import threading


def thread_body():
    s = 0
    for i in xrange(1000000000):
        s += i
    print(s)


workers = []
for id in range(6):
    worker = threading.Thread(target=thread_body)
    worker.start()
    workers.append(worker)

for worker in workers:
    worker.join()
