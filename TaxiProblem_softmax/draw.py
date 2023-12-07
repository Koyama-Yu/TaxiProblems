import numpy as np
import matplotlib.pyplot as plt

q_value = np.load('Qvalue.npy', allow_pickle=True)
print(q_value)
# print(np.max(q_value))
# print(np.min(q_value))