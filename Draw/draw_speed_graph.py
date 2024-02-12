#from operator import truediv
import numpy as np
from matplotlib import pyplot as plt
import csv

firststep_num = 100
fname = 'ave_steps_syukutai_N120_rep.npy'   #change when you need
steps = np.load(fname)
#firststep = np.zeros(firststep_num)
ave = np.mean(steps[101:])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(np.arange(0, firststep_num), steps[:100], color = 'dodgerblue')
plt.axhline(ave, ls = "-.", color = "slateblue")

#x, y軸の範囲変更
plt.xlim(0, 100)
plt.ylim(0, 2500)

plt.xlabel("episodes")
plt.ylabel("steps")
plt.savefig('first100_steps_ver2.png')   #change when you need

""" 
firststep_num = 100
firststep1 = np.zeros(firststep_num)
firststep2 = np.zeros(firststep_num)
firststep3 = np.zeros(firststep_num)
firststep4 = np.zeros(firststep_num)
for i in range(firststep_num):
    firststep1[i] = ary1[i]
    firststep2[i] = ary2[i]
    firststep3[i] = ary3[i]
    firststep4[i] = ary4[i]


plt.plot(np.arange(0, firststep_num), firststep1, color = 'red')
plt.axhline(490.13658333333365, ls = "-.", color = "magenta")
"""

