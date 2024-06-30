#参考サイト : https://www.tcom242242.net/entry/ai-2/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92/%E3%80%90%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%80%81%E5%85%A5%E9%96%80%E3%80%91q%E5%AD%A6%E7%BF%92_%E8%BF%B7%E8%B7%AF%E3%82%92%E4%BE%8B%E3%81%AB/
#main.py, Environment.pyもここから

import copy
import numpy as np
from Const import constant as cst

class QLearningAgent:
    def __init__(self, alpha=.2, tau=.1,  gamma=.99,  actions=None, observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.actions = actions
        self.state = observation   #状態
        self.ini_state = observation   #初期状態
        self.previous_state = None  #前の状態
        self.previous_action = None #前の行動
        self.q_values = self._init_q_values()   #Qテーブル(辞書型)の初期化

    def _init_q_values(self):
        q_values = {}
        #q_values[self.state] = np.repeat(0.0, len(self.actions)
        legal_act_num = self._check_legal_act_num(self.state)
        q_values[self.state] = np.repeat(0.0, legal_act_num)
        return q_values

    def init_state(self):
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self):
        #softmax行動選択
        values = self.q_values[self.state]
        max_value = np.max(values)  #オーバーフロー対策
        exp_values = np.exp((values - max_value) / self.tau)
        sum_exp_values = np.sum(exp_values)
        p = exp_values / sum_exp_values  # 確率分布の生成

        action = np.random.choice(np.arange(len(values)), p=p)  # 確率分布pに従ってランダムで選択   

        self.previous_action = action
        return action

    def observe(self, next_state):
        """
            次の状態と報酬の観測
        """
        #next_state = str(next_state)
        if next_state not in self.q_values:  # 始めて訪れる状態であれば
            legal_act_num = self._check_legal_act_num(next_state)
            #self.q_values[next_state] = np.repeat(0.0, len(self.actions))
            self.q_values[next_state] = np.repeat(0.0, legal_act_num)

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

    def learn(self, reward):
        q = self.q_values[self.previous_state][self.previous_action]  # Q(s, a)
        max_q = max(self.q_values[self.state])  # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.q_values[self.previous_state][self.previous_action] = q + \
            (self.alpha * (reward + (self.gamma * max_q) - q))
        #print(self.previous_action, self.previous_state)   #デバッグ用
    
    def save_q(self, fname):
        np.save(fname, self.q_values)
    
    def _check_legal_act_num(self, check_state):
        if check_state == cst.const.state_set['City B']:
            legal_act_num = len(cst.const.state_set) - 1
        else:
            legal_act_num = len(cst.const.state_set)
        return legal_act_num