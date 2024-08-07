#参考サイト : https://www.tcom242242.net/entry/ai-2/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92/%E3%80%90%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%80%81%E5%85%A5%E9%96%80%E3%80%91q%E5%AD%A6%E7%BF%92_%E8%BF%B7%E8%B7%AF%E3%82%92%E4%BE%8B%E3%81%AB/
#main.py, Environment.pyもここから

import copy
import numpy as np
from Const import constant as cst
from BetaAutomata.BetaAutomata import *

class QLearningAgent:
    def __init__(self, alpha=.2, epsilon=.1,  gamma=.99,  actions=None, observation=None, la_state_num=20, min_x=-1, max_x=10, theta=1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.state = observation   #状態
        self.ini_state = observation   #初期状態
        self.previous_state = None  #前の状態
        self.previous_action = None #前の行動
        self.q_values = self._init_q_values()   #Qテーブル(辞書型)の初期化
        self.la_state_num = la_state_num
        self.min_x = min_x
        self.max_x = max_x
        self.automaton = self._init_automaton()
        self.la_select_action = None
        self.theta = theta
        

    def _init_q_values(self):
        q_values = {}
        #q_values[self.state] = np.repeat(0.0, len(self.actions)
        legal_act_num = self._check_legal_act_num(self.state)
        q_values[self.state] = np.repeat(0.0, legal_act_num)
        return q_values
    
    def _init_automaton(self):
        automaton = {}
        legal_act_num = self._check_legal_act_num(self.state)
        automaton[self.state] = BetaAutomata(self.la_state_num, legal_act_num, self.min_x, self.max_x, self.actions[:legal_act_num])
        return automaton

    def init_state(self):
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self):
        # # ε-greedy選択
        # if np.random.uniform() < self.epsilon:  # random行動
        #     action = np.random.randint(0, len(self.q_values[self.state]))
        #     #print('random')
        # else:   # greedy 行動
        #     action = np.argmax(self.q_values[self.state])
        #     #print('greedy')

        # self.previous_action = action
        # return action
        myu_t, self.la_select_action = self.automaton[self.state].get_myu_t_and_action()
        self.previous_action = self.la_select_action
        return self.la_select_action

    def observe(self, next_state):
        """
            次の状態と報酬の観測
        """
        #next_state = str(next_state)
        if next_state not in self.q_values:  # 始めて訪れる状態であれば
            legal_act_num = self._check_legal_act_num(next_state)
            #self.q_values[next_state] = np.repeat(0.0, len(self.actions))
            self.q_values[next_state] = np.repeat(0.0, legal_act_num)
            self.automaton[next_state] = BetaAutomata(self.la_state_num, legal_act_num, self.min_x, self.max_x, self.actions[:legal_act_num]) #オートマトンのテーブルに追加

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

    def learn(self, reward):
        q = self.q_values[self.previous_state][self.previous_action]  # Q(s, a)
        max_q = max(self.q_values[self.state])  # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.q_values[self.previous_state][self.previous_action] = q + \
            (self.alpha * (reward + (self.gamma * max_q) - q))
        #print(self.previous_action, self.previous_state)   #デバッグ用
        self.automaton[self.previous_state].renew_lamda_of_BE_ko(self.q_values[self.previous_state][self.previous_action], self.la_select_action, self.theta)
    
    def save_q(self, fname):
        np.save(fname, self.q_values)
    
    def _check_legal_act_num(self, check_state):
        if check_state == cst.const.state_set['City B']:
            legal_act_num = len(cst.const.state_set) - 1
        else:
            legal_act_num = len(cst.const.state_set)
        return legal_act_num