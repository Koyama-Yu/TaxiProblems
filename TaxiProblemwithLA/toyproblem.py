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
        epsilon=cst.const.EPSILON,
        actions=cst.const.ACTIONS,
        observation=ini_state
    )
    accum_rewards = []    #累積報酬の保存
    mean_rewards = []     #平均報酬の保存
    all_episode_states = []     #移った状態の保存
    #all_episode_actions = []    #選択された行動の保存
    
    #実験
    for episode in range(cst.const.NB_EPISODE):
        print(f"episode:{episode}")
        episode_reward = []    #そのエピソードの累積報酬
        episode_states = []    #そのエピソードの状態
        for step in range(cst.const.NB_STEP):    
            action = agent.act()    #行動選択
            state, reward = env.step(action)
            agent.observe(state)    #状態の観測
            agent.learn(reward)             #学習
            episode_reward.append(reward)
            episode_states.append(action)
        accum_rewards.append(np.sum(episode_reward))    #エピソードの累積報酬  
        mean_rewards.append(np.mean(episode_reward))    #エピソードの平均報酬
        all_episode_states.append(episode_states)       #エピソードの状態（遷移）
        state = env.reset()
        agent.observe(state)    #初期状態に
        
    agent.save_q('Qvalue.npy')
    # pop_prob = []
    # unpop_prob = []
    # for episode_state in all_episode_states:
    #     pop_count = episode_state.count(cst.const.state_set['Pop'])
    #     unpop_count = episode_state.count(cst.const.state_set['Unpop'])
    #     pop_prob.append(pop_count / cst.const.NB_STEP)
    #     unpop_prob.append(unpop_count / cst.const.NB_STEP)
        
    # #結果のプロット
    # #累積報酬
    # plt.plot(np.arange(NB_EPISODE), accum_rewards)
    # plt.xlabel("episode")
    # plt.ylabel("accum_reward")
    # plt.savefig("accum_result.jpg")
    # #平均報酬
    # plt.plot(np.arange(cst.const.NB_EPISODE), mean_rewards)
    # plt.xlabel("episode")
    # plt.ylabel("mean_reward")
    # plt.savefig("mean_result.jpg") 
    
    # plt.plot(np.arange(cst.const.NB_EPISODE), pop_prob)
    # plt.xlabel("episode")
    # plt.ylabel("pop_prob")
    # plt.savefig("./pop_prob.jpg") 
    
    # plt.plot(np.arange(cst.const.NB_EPISODE), unpop_prob)
    # plt.xlabel("episode")
    # plt.ylabel("unpop_prob")
    # plt.savefig("./unpop_prob.jpg") 
    #plt.show()   