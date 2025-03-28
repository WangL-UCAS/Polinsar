import numpy as np
import h5py
import math

def make_pauli(array_HH, array_VV, array_HV, row, col, filename_k1="pauli_k1.h5", filename_k1_T="pauli_k1_T.h5"):
    # 计算 k11, k22, k33
    sqrt2 = math.sqrt(2)
    k11 = (array_HH + array_VV) / sqrt2
    k22 = (array_HH - array_VV) / sqrt2
    k33 = (array_HV * 2) / sqrt2

    # 构建 k1 和 k1_T（也就是共轭转置）
    k1 = np.stack((k11, k22, k33), axis=-1)  # shape: (row, col, 3)
    k1 = k1[..., np.newaxis]  # shape: (row, col, 3, 1)
    k1_T = np.conj(k1).swapaxes(-1, -2)  # shape: (row, col, 1, 3)

    # 保存 k1
    with h5py.File(filename_k1, 'w') as f:
        f.create_dataset('k1', data=k1, dtype=k1.dtype)
    print(f"------- Pauli 分解结束，k1 结果已保存至 {filename_k1} ----------")

    # 保存 k1_T
    with h5py.File(filename_k1_T, 'w') as f:
        f.create_dataset('k1_T', data=k1_T, dtype=k1_T.dtype)
    print(f"------- Pauli 分解结束，k1_T 结果已保存至 {filename_k1_T} ----------")

    return k1, k1_T
