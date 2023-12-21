import numpy as np
import matplotlib.pyplot as plt

theta = 10
q_value = np.load(f'./Qvalues/Qvalue_la_theta{theta}.npy', allow_pickle=True)
print(q_value)