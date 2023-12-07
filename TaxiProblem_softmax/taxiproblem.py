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
        tau=cst.const.TAU,
        actions=cst.const.ACTIONS,
        observation=ini_state
    )
    
    opt_act_num = 0
    opt_act_prob = []
    #実験
    for episode in range(cst.const.NB_EPISODE):
        print(f"episode:{episode}")
        for step in range(cst.const.NB_STEP):    
            action = agent.act()    #行動選択
            if action == cst.const.act_set['Stand']:
                opt_act_num = opt_act_num + 1
            state, reward = env.step(action)
            agent.observe(state)    #状態の観測
            agent.learn(reward)             #学習
        opt_act_prob.append(opt_act_num / cst.const.NB_STEP)
        opt_act_num = 0
        state = env.reset()
        agent.observe(state)    #初期状態に
        
    agent.save_q('Qvalue_softmax.npy')
    #結果のプロット
    plt.plot(np.arange(cst.const.NB_EPISODE), np.array(opt_act_prob))
    plt.xlabel("episode")
    plt.ylabel("optimal action times")
    plt.savefig("opt_act_result_softmax.jpg")
    plt.show()