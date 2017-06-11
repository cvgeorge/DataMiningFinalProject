import sys
import re
import math
import copy
import time

def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

(train, varnames) = read_data("train_mush.csv")
for row in train:
	time.sleep(1)
	print row[-1]