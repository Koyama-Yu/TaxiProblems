#βタイプオートマトン

import numpy as np

class BE:
    def __init__(self, act_num, state_num, min_x, max_x):
        self.min_x = min_x  #応答値の最小値 x_
        self.max_x = max_x  #応答値の最大値 ￣x
        self.lamda = np.full(state_num, 1/act_num)    #状態確率ベクトル λ
        #self.omega = np.arange(1, state_num)   #状態集合 Ω
        self.myu = np.arange(1, state_num + 1) / (state_num + 1)   #出力集合 μ

# public method
    def get_myu(self):   #λに従って出力値 μ_kiを出力
        prov = self.lamda / sum(self.lamda)
        return np.random.choice(self.myu, p=prov)
    
    def renew_lamda(self, resp_x, gamma): #最適行動を選んだBEの状態確率の更新
        self.__calc_new_lamda(resp_x, gamma)
        self.__nomalization_lamda()
    
#private method
    def __calc_new_lamda(self, resp_x, gamma):  #各ラムダを更新
        x = (resp_x - self.min_x) / (self.max_x - self.min_x)
        self.lamda = self.lamda * (self.myu**x * (1-self.myu) ** (1-x)) ** gamma
    
    def __nomalization_lamda(self): #正規化
        self.lamda = self.lamda / self.lamda.sum()