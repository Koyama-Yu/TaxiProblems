import numpy as np
import matplotlib.pyplot as plt

theta = 1
q_value = np.load(f'./Qvalues/Qvalue_la_theta{theta}.npy', allow_pickle=True)
act_prob = np.load(f'./probabilities/OptimalActionProbability_Ave_LA_theta{theta}.npy', allow_pickle=True)
#print(q_value)
print(act_prob)