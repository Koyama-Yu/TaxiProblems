"""グラフ描画用"""

#TODO ここに各ファイルを読み取って, グラフを描画するコードを記述
import numpy as np
from matplotlib import pyplot as plt
import csv

episode_num = 100
fname_theta1 = 'OptimalActionProbability_Ave_LA_theta1.npy'   #change when you need
fname_theta2 = 'OptimalActionProbability_Ave_LA_theta2.npy'   #change when you need
fname_theta5 = 'OptimalActionProbability_Ave_LA_theta5.npy'   #change when you need
fname_theta10 = 'OptimalActionProbability_Ave_LA_theta10.npy'   #change when you need

episodes_theta1 = np.load(fname_theta1)
episodes_theta2 = np.load(fname_theta2)
episodes_theta5 = np.load(fname_theta5)
episodes_theta10 = np.load(fname_theta10)
#firststep = np.zeros(firststep_num)
#ave = np.mean(steps[101:])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(np.arange(0, episode_num), episodes_theta1, ls='-', label='θ = 1', color='dodgerblue')
plt.plot(np.arange(0, episode_num), episodes_theta2, ls='--', label='θ = 2', color='dodgerblue')
plt.plot(np.arange(0, episode_num), episodes_theta5, ls=':', label='θ = 5', color='dodgerblue')
plt.plot(np.arange(0, episode_num), episodes_theta10, ls='-.', label='θ = 10', color='dodgerblue')
#plt.plot(np.arange(0, step_num - 200), steps1[:300], color = 'dodgerblue')
#plt.plot(np.arange(0, step_num - 200), steps2[:300], color = 'salmon')
#plt.axhline(ave, ls = "-.", color = "slateblue")
#plt.xlim(0, 100)
#plt.ylim(0, 1.0)
plt.xlabel("episodes")
plt.ylabel("optimal action selection rate")
plt.legend()
plt.savefig('compare_each_theta.png')   #change when you need
plt.show()