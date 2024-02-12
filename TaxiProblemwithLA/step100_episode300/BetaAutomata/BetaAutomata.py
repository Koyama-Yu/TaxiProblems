#LAのクラス
#BEを使って, 推定を行う                 ↓強化学習用のため
#やらせることは「各BEが出すμ」 のなかで<<最大>>のものを選択して、式で計算して最終的な出力を出す
#当然,行動選択もする

import numpy as np
from BetaAutomata.BE import *

class BetaAutomata:
    def __init__(self, state_num, act_num, min_x, max_x, act_set):
        self.min_x = min_x  #応答値の最小値 x_
        self.max_x = max_x  #応答値の最大値 ￣x
        self.bayes_estims = [BE(act_num, state_num, min_x, max_x) for i in range(act_num)]   #ベイズ推定器のインスタンスを生成
        self.act_set = act_set  #行動集合

# public metod
    def get_myu_t_and_action(self): #μ(t), α(t)の出力
        max_myu, max_index = self.__get_max_myu_and_max_index()
        action = self.__get_action(max_index)
        myu_t = self.__get_myu_t(max_myu)
        return myu_t, action
    
    def renew_lamda_of_BE_ko(self, resp_x, ko, gamma):  #BEkoのλを更新
        self.bayes_estims[ko].renew_lamda(resp_x, gamma)
    
    def print_lamda(self, BE_num):  #デバッグ用
        for i in range(BE_num):
            print(f"BE{i}", self.bayes_estims[i].lamda)
    
    # def save_lamda(self, BE_num):
    #     for i in range(BE_num):
    #         np.save(f"BE{i}lamda.npy", self.bayes_estims[i].lamda)
    #         #print(f"BE{i}", self.bayes_estims[i].lamda)

# private method
    def __get_max_myu_and_max_index(self):  #「各BEが出すμ」の中で<<<最大>>>のものを選択し, その番号(行動)も取得
        return max( (self.bayes_estims[i].get_myu(), i) for i in range(len(self.bayes_estims)))
    
    def __get_action(self, act_index):  #番号から行動選択
        return self.act_set[act_index]
    
    def __get_myu_t(self, myu_ko): #μ(t)を計算して出力
        return (self.max_x - self.min_x) * myu_ko + self.min_x