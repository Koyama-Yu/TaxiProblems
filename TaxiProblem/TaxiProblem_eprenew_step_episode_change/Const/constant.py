"""
定数を記述するファイル
"""
from Const import const
import numpy as np

#定数
const.STATE_NUM = 3 #状態1, 2, 3が街A, B, Cにいる状態 (プログラム上では0 or 1 or 2)
const.ACT_NUM = 3 #1:通行人探し 2:タクスタ 3:無線(同上)
const.NB_REPEAT = 50      #実験の繰り返し数
const.NB_EPISODE = 300    # エピソード数
const.NB_STEP = 100
const.EPSILON_INI = 1.    # 探索率
const.EPSILON_CO = 0.999
const.ALPHA = .1      # 学習率
const.GAMMA = .90     # 割引率
const.ACTIONS = np.arange(const.ACT_NUM)    #行動集合
const.transit_prob = np.array(
    [
        [   #時刻tで状態0
            [1/2, 1/4, 1/4],    #行動0をしたときの各状態の遷移確率
            [1/16, 3/4, 3/16],  #行動1をしたときの各状態の遷移確率
            [1/4, 1/8, 5/8]     #行動2をしたときの各状態の遷移確率
        ],
        [   #時刻tで状態1
            [1/2, 0, 1/2],      #上と似た感じ
            [1/16, 7/8, 1/16] 
        ],
        [   #時刻tで状態2
            [1/4, 1/4, 1/2],    #行動0をしたときの各状態の遷移確率
            [1/8, 3/4, 1/8],    #行動1をしたときの各状態の遷移確率
            [3/4, 1/16, 3/16]   #行動2をしたときの各状態の遷移確率
        ]
    ]
)
const.rewards = np.array(
    [
        [   #時刻tで状態0
            [10, 4, 8],     #行動0をしたときの各状態での利得
            [8, 2, 4],      #行動1をしたときの各状態での利得
            [4, 6, 4]      #行動2をしたときの各状態での利得
        ],
        [   #時刻tで状態1
            [14, 0, 18], #上と似た感じ
            [8, 16, 8] 
        ],
        [   #時刻tで状態2
            [10, 2, 8],     #行動0をしたときの各状態での利得
            [6, 4, 2],      #行動1をしたときの各状態での利得
            [4, 0, 8]      #行動2をしたときの各状態での利得
        ]
    ]
)
const.state_set = {'City A':0, 'City B':1, 'City C':2} 
const.act_set = {'Find':0, 'Stand':1, 'Call':2}
const.start_state = const.state_set['City A']
