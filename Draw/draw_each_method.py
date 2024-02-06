"""グラフ描画用"""

#TODO ここに各ファイルを読み取って, グラフを描画するコードを記述
import numpy as np
from matplotlib import pyplot as plt
import csv

episode_num = 100
fname_eps_fix = 'OptimalActionProbability_Ave_Epsilon01.npy'   #change when you need
fname_eps_ren = 'OptimalActionProbability_Ave_Epsilon_Co0.999.npy'   #change when you need
fname_la = 'OptimalActionProbability_Ave_LA_theta1.npy'   #change when you need
fname_softmax = 'OptimalActionProbability_Ave_Softmax_t8.0.npy'   #change when you need

episodes_eps_fix = np.load(fname_eps_fix)
episodes_eps_ren = np.load(fname_eps_ren)
episodes_la = np.load(fname_la)
episodes_softmax = np.load(fname_softmax)
#firststep = np.zeros(firststep_num)
#ave = np.mean(steps[101:])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(np.arange(0, episode_num), episodes_eps_fix, ls='--', label='ε-greedy (ε fixed)', color='dodgerblue')
plt.plot(np.arange(0, episode_num), episodes_eps_ren, ls=':', label='ε-greedy (ε renewed)', color='dodgerblue')
plt.plot(np.arange(0, episode_num), episodes_la, ls='-',label='LQ', color='dodgerblue')
plt.plot(np.arange(0, episode_num), episodes_softmax, ls='-',label='softmax (T = 8.0)', color='salmon')
#plt.plot(np.arange(0, step_num - 200), steps1[:300], color = 'dodgerblue')
#plt.plot(np.arange(0, step_num - 200), steps2[:300], color = 'salmon')
#plt.axhline(ave, ls = "-.", color = "slateblue")
#plt.xlim(0, 100)
#plt.ylim(0, 1.0)
plt.xlabel("episodes")
plt.ylabel("optimal action selection rate")
plt.legend()
plt.savefig('compare_each_method_add_softmax.png')   #!change
plt.show()