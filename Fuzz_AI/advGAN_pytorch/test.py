from progress.bar import Bar
import time

a = 0

bar = Bar('Processing', max=20)
for i in range(20):
    # Do some work
    a += 100
    time.sleep(0.2)
    print('\t', a)
    bar.next()

bar.finish()

print(a)
