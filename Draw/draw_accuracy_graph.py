#from operator import truediv
import numpy as np
from matplotlib import pyplot as plt
import csv

step_num = 500
fname1 = 'ave_steps_original.npy'   #change when you need
fname2 = 'ave_steps_syukutai_N120_rep.npy'   #change when you need
steps1 = np.load(fname1)
steps2 = np.load(fname2)
#firststep = np.zeros(firststep_num)
#ave = np.mean(steps[101:])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(np.arange(0, step_num), steps1[:500], color = 'dodgerblue')
plt.plot(np.arange(0, step_num), steps2[:500], color = 'salmon')
#plt.plot(np.arange(0, step_num - 200), steps1[:300], color = 'dodgerblue')
#plt.plot(np.arange(0, step_num - 200), steps2[:300], color = 'salmon')
#plt.axhline(ave, ls = "-.", color = "slateblue")
plt.xlim(0, 500)
plt.ylim(0, 2500)
plt.xlabel("episodes")
plt.ylabel("steps")
plt.savefig('accuracy_graph.png')   #change when you need

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

