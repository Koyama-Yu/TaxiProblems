"""タクシー問題(Taxi-Problem)シミュレーション"""

import numpy as np
import matplotlib.pyplot as plt
from qLearningAgent import QLearningAgent
from environment import Environment
from Const import constant as cst

if __name__ == '__main__':
    env = Environment(cst.const.STATE_NUM, 
        cst.const.start_state,   #スタートする状態
        cst.const.transit_prob,
        cst.const.rewards)
    ini_state = env.start_state
    
    opt_act_num = 0
    opt_act_prob_ave = np.zeros(cst.const.NB_EPISODE)
    #実験
    #TODO 並列処理を後に入れる
    for repeat in range(cst.const.NB_REPEAT):
        print(f"repeat:{repeat}")
        agent = QLearningAgent(
        alpha=cst.const.ALPHA,
        gamma=cst.const.GAMMA,
        tau=cst.const.TAU,
        actions=cst.const.ACTIONS,
        observation=ini_state
        )
        for episode in range(cst.const.NB_EPISODE):
            #print(f"episode:{episode}")
            for step in range(cst.const.NB_STEP):    
                action = agent.act()    #行動選択
                if action == cst.const.act_set['Stand']:
                    opt_act_num = opt_act_num + 1
                state, reward = env.step(action)
                agent.observe(state)    #状態の観測
                agent.learn(reward)     #学習
            opt_act_prob_ave[episode] = opt_act_prob_ave[episode] + (opt_act_num / cst.const.NB_STEP)
            opt_act_num = 0
            state = env.reset()
            agent.observe(state)    #初期状態に
    opt_act_prob_ave = opt_act_prob_ave / cst.const.NB_REPEAT
    agent.save_q('Qvalue_softmax.npy')  #! ファイルの名前は適宜変更
    np.save('OptimalActionProbability_Ave_Softmax.npy', opt_act_prob_ave)   #! ファイルの名前は適宜変更
    
    #結果のプロット
    plt.plot(np.arange(cst.const.NB_EPISODE), np.array(opt_act_prob_ave))
    plt.xlabel("episode")
    plt.ylabel("optimal action times")
    plt.savefig("opt_act_result_softmax.jpg")
    plt.show()