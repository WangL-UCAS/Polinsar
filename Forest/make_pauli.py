import os
import numpy as np
import h5py
import math


def make_pauli(array_HH, array_VV, array_HV, filename_k, filename_k_T, row, col, name_k, name_k_T):
    sqrt2 = math.sqrt(2)
    k11 = (array_HH + array_VV) / sqrt2
    k22 = (array_HH - array_VV) / sqrt2
    k33 = (array_HV * 2) / sqrt2

    k1 = np.stack((k11, k22, k33), axis=-1)  # shape: (row, col, 3)
    k1 = k1[..., np.newaxis]  # shape: (row, col, 3, 1)
    k1_T = np.conj(k1).swapaxes(-1, -2)  # shape: (row, col, 1, 3)

    # 获取目录路径
    dir_k = os.path.dirname(filename_k)
    dir_k_T = os.path.dirname(filename_k_T)

    # 确保目录存在
    os.makedirs(dir_k, exist_ok=True)
    os.makedirs(dir_k_T, exist_ok=True)

    # 保存 k1
    with h5py.File(filename_k, 'w') as f:
        f.create_dataset(name_k, (row,col,3,1),data=k1,dtype=k1.dtype)
    print(f"✅ Pauli 分解结束，k1 结果已保存至 {filename_k}")

    # 保存 k1_T
    with h5py.File(filename_k_T, 'w') as f:
        f.create_dataset(name_k_T, (row,col,1,3),data=k1_T, dtype=k1_T.dtype)
    print(f"✅ Pauli 分解结束，k1_T 结果已保存至 {filename_k_T}")

    return k1, k1_T
