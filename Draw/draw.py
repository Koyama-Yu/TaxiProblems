"""グラフ描画用"""

#TODO ここに各ファイルを読み取って, グラフを描画するコードを記述
import numpy as np
from matplotlib import pyplot as plt
import csv

episode_num = 100
fname_eps = 'OptimalActionProbability_Ave_Epsilon01.npy'   #change when you need
fname_la = 'OptimalActionProbability_Ave_LA_theta1.npy'   #change when you need
episodes_eps = np.load(fname_eps)
episodes_la = np.load(fname_la)
#firststep = np.zeros(firststep_num)
#ave = np.mean(steps[101:])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(np.arange(0, episode_num), episodes_eps, ls='--', label='ε-greedy', color='dodgerblue')
plt.plot(np.arange(0, episode_num), episodes_la, ls='-',label='LQ', color='dodgerblue')
#plt.plot(np.arange(0, step_num - 200), steps1[:300], color = 'dodgerblue')
#plt.plot(np.arange(0, step_num - 200), steps2[:300], color = 'salmon')
#plt.axhline(ave, ls = "-.", color = "slateblue")
#plt.xlim(0, 100)
#plt.ylim(0, 1.0)
plt.xlabel("episodes")
plt.ylabel("optimal action selection rate")
plt.legend()
plt.savefig('compare_each_method.png')   #change when you need
plt.show()