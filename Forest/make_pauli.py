
import numpy as np
import math


def make_pauli(array_HH, array_VV, array_HV,row,col):
    ## 获取卫星影像的像元数量
    k1 = np.zeros((row,col,3,1),dtype=complex)
    for i in range(row):
        for j in range(col):
            k11 = (array_HH[i,j] + array_VV[i,j]) / math.sqrt(2)
            k22 = (array_HH[i,j] - array_VV[i,j]) / math.sqrt(2)
            k33 = (array_HV[i,j] * 2) / math.sqrt(2)
            k1[i,j] = np.array([[k11],[k22],[k33]])
    print("-------pauli 分解结束----------")
    return k1

# 计算 Pauli 分解

## 计算共轭转置，也就是k1_T  k2_T
def make_pauli_T(array_HH, array_VV, array_HV,row,col):
    k1 = np.zeros((row,col,1,3),dtype=complex)
    for i in range(row):
        for j in range(col):
            k11 = (array_HH[i,j] + array_VV[i,j]) / math.sqrt(2)
            k22 = (array_HH[i,j] - array_VV[i,j]) / math.sqrt(2)
            k33 = (array_HV[i,j] * 2) / math.sqrt(2)
            k = np.array([k11,k22,k33])
            k_conj = np.conj(k)
            k1[i,j] = np.array(k_conj)
    print("-------pauli 分解结束----------")
    return k1
