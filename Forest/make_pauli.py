import os.path
import h5py
import numpy as np
import math

def make_pauli(array_HH, array_VV, array_HV, filename_k,filename_k_T,row, col,name_k,namme_k_T):
    # 计算k11, k22, k33的矩阵运算
    sqrt2 = math.sqrt(2)
    k11 = (array_HH + array_VV) / sqrt2
    k22 = (array_HH - array_VV) / sqrt2
    k33 = (array_HV * 2) / sqrt2

    # 构建k1和k1_T矩阵
    k1 = np.stack((k11, k22, k33), axis=-1)  # shape: (row, col, 3)
    k1 = k1[..., np.newaxis]  # 添加最后一个维度，变为 (row, col, 3, 1)

    # 共轭转置
    k1_T = np.conj(k1).swapaxes(-1, -2)
    k1 = k1.astype(np.complex64)
    k1_T = k1_T.astype(np.complex64)
    print(k1_T.shape)
    print()
    dir_k = os.path.dirname(filename_k)
    dir_k_T = os.path.dirname(filename_k_T)

    os.makedirs(dir_k,exist_ok=True)
    os.makedirs(dir_k_T,exist_ok=True)

    with h5py.File(filename_k,'w') as f:
        f.create_dataset(name_k,(row,col,3,1),data=k1,dtype=k1.dtype)
        print("已保存k")
    with h5py.File(filename_k_T,'w') as f:
        f.create_dataset(namme_k_T,(row,col,1,3),data=k1_T,dtype=k1_T.dtype)
        print("k转置共轭计算结束")
    print("-------pauli 分解结束----------")
    return k1,k1_T
# 计算 Pauli 分解

##
