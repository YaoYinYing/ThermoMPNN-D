import os

import matplotlib.pyplot as plt

d = 'ProteinMPNN_000_0/'
f = 'log.txt'

ftot = os.path.join(d, f)

with open(ftot) as fopen:
    lines = fopen.readlines()

val = [float(a.split(' ')[-1].strip('\n')) for a in lines[1:]]
ep = [int(a.split(' ')[1].strip(',')) for a in lines[1:]]
print(val, ep)


plt.scatter(ep, val)
plt.show()
