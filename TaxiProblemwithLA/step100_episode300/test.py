import numpy as np
import os
from multiprocessing import cpu_count

dict = {'a':1, 'b':2}
print(np.random.choice(list(dict)))

print(np.random.choice(np.arange(2)))

if __name__ == "__main__":
    #print("Number of CPU cores: ", cpu_count())
    print("Number of CPU cores: ", os.cpu_count())