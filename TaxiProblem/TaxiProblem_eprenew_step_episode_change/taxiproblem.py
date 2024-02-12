"""おもちゃ屋問題(Toy-Problem)シミュレーション"""

import random
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
    agent = QLearningAgent(
        alpha=cst.const.ALPHA,
        gamma=cst.const.GAMMA,
        epsilon_ini=cst.const.EPSILON_INI,
        epsilon_co=cst.const.EPSILON_CO,
        actions=cst.const.ACTIONS,
        observation=ini_state
    )
    
    opt_act_num = 0
    opt_act_prob_ave = np.zeros(cst.const.NB_EPISODE)
    #実験
    #TODO 並列処理を後に入れる
    for repeat in range(cst.const.NB_REPEAT):
        print(f"repeat:{repeat}")
        agent = QLearningAgent(
        alpha=cst.const.ALPHA,
        gamma=cst.const.GAMMA,
        epsilon_ini=cst.const.EPSILON_INI,
        epsilon_co=cst.const.EPSILON_CO,
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
    #Q値保存
    agent.save_q(f'./Qvalues/Qvalue_epsilon_renew{cst.const.EPSILON_CO}.npy')
    np.save(f'./probabilities/OptimalActionProbability_Ave_Epsilon_Co{cst.const.EPSILON_CO}.npy', opt_act_prob_ave)
    
    #結果のプロット
    plt.plot(np.arange(cst.const.NB_EPISODE), np.array(opt_act_prob_ave))
    plt.xlabel("episode")
    plt.ylabel(f"optimal action times (k = {cst.const.EPSILON_CO})")
    plt.savefig(f"./graphs/opt_act_result_epsilon_renew{cst.const.EPSILON_CO}.jpg")
    plt.show()