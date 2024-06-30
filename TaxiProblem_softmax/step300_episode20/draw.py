import numpy as np
import matplotlib.pyplot as plt

q_value = np.load('Qvalue_softmax.npy', allow_pickle=True)
print(q_value)