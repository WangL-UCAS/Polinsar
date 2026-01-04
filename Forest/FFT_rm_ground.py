import h5py

import numpy as np

import matplotlib.pyplot as plt

with h5py.File('FP-2-927-1005-Y_max.h5','r') as f:
    data = f['FP-2-927-1005-Y_max'][:]
plt.matshow(np.abs(data),cmap='hsv')
plt.colorbar();
plt.title('No_Range_Filter')
plt.show()
