from __future__ import print_function

from multiprocessing import Process


def wrkr(p1):
	print("Running thread: " + str(p1))


maxItem = 100
curItem = 0

while (curItem < maxItem):
	t = Process(target=wrkr, kwargs=(curItem,))
	t.start()
	t.join()
	curItem += 1


