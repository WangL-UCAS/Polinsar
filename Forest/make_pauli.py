
import numpy as np
import math

def make_pauli(array_HH, array_VV, array_HV, row, col):
    # 计算k11, k22, k33的矩阵运算
    sqrt2 = math.sqrt(2)
    k11 = (array_HH + array_VV) / sqrt2
    k22 = (array_HH - array_VV) / sqrt2
    k33 = (array_HV * 2) / sqrt2

    # 构建k1和k1_T矩阵
    k1 = np.stack((k11, k22, k33), axis=-1)  # shape: (row, col, 3)
    k1 = k1[..., np.newaxis]  # 添加最后一个维度，变为 (row, col, 3, 1)

    k1_T = np.conj(k1)  # 共轭复数
    k1_T = np.swapaxes(k1_T, -1, -2)  # 转置最后两个维度，变为 (row, col, 1, 3)

    print("-------pauli 分解结束----------")
    return k1, k1_T
# 计算 Pauli 分解

##
