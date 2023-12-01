import copy
import numpy as np

class Environment:

    def __init__(self, state_num, start_state, transit_prob, rewards):
        self.state_num = state_num
        self.start_state = start_state
        self.transit_prob = transit_prob
        self.rewards = rewards
        self.agent_state = copy.deepcopy(self.start_state)  # エージェントの状態

    def step(self, action):
        """
            行動の実行
            状態, 報酬を返却
        """     
        pre_state = self.agent_state
        self.agent_state = self._compute_state(pre_state, action)
        reward = self._compute_reward(pre_state, action)
        return self.agent_state, reward
    
    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state
    
    def _compute_state(self, pre_state, action):
        p = self.transit_prob[pre_state][action] #状態遷移確率
        #状態遷移確率に従って, 遷移した状態を返す
        state = np.random.choice(np.arange(self.state_num),  p=p)
        return state
    
    def _compute_reward(self, pre_state, action):
        return self.rewards[pre_state][action][self.agent_state]
