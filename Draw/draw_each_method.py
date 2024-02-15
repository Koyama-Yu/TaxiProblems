"""グラフ描画用"""

#TODO ここに各ファイルを読み取って, グラフを描画するコードを記述
import numpy as np
from matplotlib import pyplot as plt
import csv

episode_num = 100
fname_eps_fix = './each_method/OptimalActionProbability_Ave_Epsilon01.npy'   #change when you need
fname_eps_ren = './each_method/OptimalActionProbability_Ave_Epsilon_Co0.999.npy'   #change when you need
fname_la = './each_method/OptimalActionProbability_Ave_LA_theta1.npy'   #change when you need
#fname_softmax = './each_method/OptimalActionProbability_Ave_Softmax_t8.0.npy'   #change when you need

episodes_eps_fix = np.load(fname_eps_fix)
episodes_eps_ren = np.load(fname_eps_ren)
episodes_la = np.load(fname_la)
#episodes_softmax = np.load(fname_softmax)
#firststep = np.zeros(firststep_num)
#ave = np.mean(steps[101:])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(np.arange(0, episode_num), episodes_eps_fix, ls=':', label='ε-greedy (ε 固定法)', color='royalblue', linewidth=2.0)
plt.plot(np.arange(0, episode_num), episodes_eps_ren, ls='--', label='ε-greedy (ε 更新法)', color='royalblue', linewidth=2.0)
plt.plot(np.arange(0, episode_num), episodes_la, ls='-',label='LQ', color='royalblue', linewidth=2.0)
#plt.plot(np.arange(0, episode_num), episodes_softmax, ls='-',label='softmax (T = 8.0)', color='tomato', linewidth=2.0)
#plt.plot(np.arange(0, step_num - 200), steps1[:300], color = 'dodgerblue')
#plt.plot(np.arange(0, step_num - 200), steps2[:300], color = 'salmon')
#plt.axhline(ave, ls = "-.", color = "slateblue")
#plt.xlim(0, 100)
#plt.ylim(0, 1.0)
plt.xlabel("episodes")
plt.ylabel("optimal action selection rate")
plt.legend(prop={'family':'MS Gothic', 'size':14})
plt.savefig('compare_each_method_jpn_ms.png')   #!change
plt.show()