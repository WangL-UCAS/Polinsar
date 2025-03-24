import cmath

import numpy as np
from osgeo import gdal
import math
import numpy.linalg as linalg
import time

"""
    data_ : 是指的卫星数据的日期，例如20240829；
    number_id 指的是卫星标号：例如105058528
    envi配准后的结果标识：
    saocom_20240829_105058528_QS6_D_HH_slc_rsp

"""
## ———————————— 这里是读取数据的函数————————————————
def read_gdal_array(filepath):
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset.GetRasterBand(1).ReadAsArray()

def read_data_from_file(base_file, file_head, file_mid, file_end, polarizations, date_, number_id):
    data_dict = {}  # 用字典存储数据

    for pol in polarizations:
        filepath = f"{base_file}/{file_head}{date_}_{number_id}{file_mid}{pol}{file_end}"
        print("检验数据路线是否正确：", filepath)
        key = f"array_{pol}_{date_}"  # 动态键名
        data_dict[key] = read_gdal_array(filepath)

    return data_dict  # 返回字典

#构建路径文件
file_head = "saocom_"
file_mid = "_QS6_D_"
file_end = "_slc_rsp"
Polarization = ["HH","HV","VH","VV"]
data_matrices = read_data_from_file("830", file_head, file_mid, file_end, Polarization, 20240829, 105058528)

array_HH_830 = data_matrices["array_HH_20240829"]
array_VV_830 = data_matrices["array_VV_20240829"]
array_HV_830 = data_matrices["array_HV_20240829"]
array_VH_830 = data_matrices["array_VH_20240829"]

# # 计算一下 HV 和 VH 的平均值，后续使用就只需要采取平均值了
array_HV_830 = (array_HV_830 + array_VH_830) / 2

data_matrices = read_data_from_file("909",file_head,file_mid,file_end, Polarization, 20240906, 105435075)

array_HH_909 = data_matrices["array_HH_20240906"]
array_VV_909 = data_matrices["array_VV_20240906"]
array_HV_909 = data_matrices["array_HV_20240906"]
array_VH_909 = data_matrices["array_VH_20240906"]

# 计算  HV 与 VH 两种极化方式的平均数据，后续可以采用平均值计算
array_HV_909 = (array_HV_909 + array_VH_909) / 2

#获取影像行列数
row,col = array_HH_830.shape
"""
    make_pauli 是创建pauli散射的 K 矩阵,
    md这里的计算都是像素为单位的，避免出错
"""
def make_pauli(array_HH, array_VV, array_HV):
    ## 获取卫星影像的像元数量
    k1 = np.zeros((row,col,3,1),dtype=complex)
    for i in range(row):
        for j in range(col):
            k11 = (array_HH[i,j] + array_VV[i,j]) / math.sqrt(2)
            k22 = (array_HH[i,j] - array_VV[i,j]) / math.sqrt(2)
            k33 = (array_HV[i,j] * 2) / math.sqrt(2)
            k1[i,j] = np.array([[k11],[k22],[k33]])
    return k1

# 计算 Pauli 分解
array_830_k1 = make_pauli(array_HH_830, array_VV_830, array_HV_830)
array_909_k2 = make_pauli(array_HH_909, array_VV_909, array_HV_909)
print("-------pauli 分解结束")

## 计算共轭转置
def make_pauli_T(array_HH, array_VV, array_HV):
    k1 = np.zeros((row,col,1,3),dtype=complex)
    for i in range(row):
        for j in range(col):
            k11 = (array_HH[i,j] + array_VV[i,j]) / math.sqrt(2)
            k22 = (array_HH[i,j] - array_VV[i,j]) / math.sqrt(2)
            k33 = (array_HV[i,j] * 2) / math.sqrt(2)
            k = np.array([k11,k22,k33])
            k_conj = np.conj(k)
            k1[i,j] = np.array(k_conj)
    return k1

array_830_k1_T = make_pauli_T(array_HH_830, array_VV_830, array_HV_830)
array_909_k2_T = make_pauli_T(array_HH_909, array_VV_909, array_HV_909)

print("-------pauli以及共轭计算计算结束-------")

"""
    ------- 计算 T6 (主要是T11、T22、Omaga12)并保存---------
"""
def make_T6(array_pauli_mask,array_pauli_T_mask,array_pauli_slave,array_pauli_T_slave):

    """
    计算T6矩阵中的 T11 T22 以及 Ω12
    mask是主影像，slave是副影像， T表示转置共轭结果
    mask得到是K1  slave 是k2
    """
    T11 = np.zeros((row,col,3,3),dtype=complex)
    T22 = np.zeros((row,col,3,3),dtype=complex)
    Omaga12 = np.zeros((row,col,3,3),dtype=complex)

    for i in range(row):
        for j in range(col):
            T11[i,j] = array_pauli_mask[i,j]@array_pauli_T_mask[i,j]
            T22[i,j] = array_pauli_slave[i,j]@array_pauli_T_slave[i,j]
            Omaga12[i,j] = array_pauli_mask[i,j]@array_pauli_T_slave[i,j]

    return T11,T22,Omaga12


T11 = np.zeros((row,col,3,3),dtype=complex)
T22 = np.zeros((row,col,3,3),dtype=complex)
Omaga12 = np.zeros((row,col,3,3),dtype=complex)

T11 , T22 , Omaga12 = make_T6(array_830_k1 , array_830_k1_T , array_909_k2, array_909_k2_T)

print("---------计算T11 T22 O12 结果结束-----------")


"""
    计算相干性优  copy的kapok代码，调试使用
    make_MAX 计算相干性优秀，WangL编写，不是调用kapok代码
    采用的是拉格朗日计算最优
"""
def make_MAX(T11,T22,Omaga12):

    """
    :param T11:
    :param T22:
    :param Omaga12:
    :return: Y_MAX, Y_MID, Y_END
    """
    T11_inv = np.linalg.inv(T11)
    T22_inv = np.linalg.inv(T22)

    # 三个复相干系数
    Y_MAX = np.array((row,col),dtype=complex)
    Y_MID = np.array((col,row),dtype=complex)
    Y_END = np.array((col,row),dtype=complex)
    for i in range(row):
        for j in range(col):
            #构造特征矩阵
            Matrix1 = T22_inv[i,j] @ Omaga12[i,j].conj().T @ T11_inv[i,j] @ Omaga12[i,j].conj().T
            Matrix2 = T11_inv[i,j] @ Omaga12 @ T22_inv[i,j] @ Omaga12[i,j].conj().T

            #计算特征值和特征向量  np.linalg.eig 方法会返回两个值，第一个返回值是矩阵的特征值，第二个返回值是矩阵的特征向量，Matrix1 和 Matrix2 是共轭关系，因此这两个矩阵的特征值是相等的，取其中三个就行了
            eig_values1, eig_vectors1 = np.linalg.eig(Matrix1)
            eig_values2, eig_vectors2 = np.linalg.eig(Matrix2)

            sorted_indices = np.argsort(eig_values1)[::-1]  # 从大到小排序索引,注意这里的返回值sorted_indices 是索引而不是数值，因此在这可以通过索引获取实际的特征值和特征向量方便后续计算
            """
            下面的计算就是要将v保存三个row ，col 的矩阵，还要计算相位归一化，因为要确定特征向量，参考最新的谷歌GPT搜索，因为有三个优复相干系数，
            eig_vectors1_MAX_T 表示v_max对应的w1特征向量
            Angle_max表示v_max 对应的特征向量w1 与 w2 计算归一化时的复度角
            """
            #  计算相位归一化
            Angle_max = cmath.phase(eig_vectors1[sorted_indices[0]].conj().T * eig_vectors2[sorted_indices[0]])
            Angle_mid = cmath.phase(eig_vectors1[sorted_indices[1]].conj().T * eig_vectors2[sorted_indices[1]])
            Angle_end = cmath.phase(eig_vectors1[sorted_indices[2]].conj().T * eig_vectors2[sorted_indices[2]])

            V_max = math.sqrt(eig_values1[sorted_indices[0]])
            V_mid = math.sqrt(eig_values1[sorted_indices[1]])
            V_end = math.sqrt(eig_values1[sorted_indices[2]])

            #  计算复相干系数
            Y_MAX[i,j] = V_max * math.exp(-1j * Angle_max)
            Y_MID[i,j] = V_mid * math.exp(-1j * Angle_mid)
            Y_END[i,j] = V_end * math.exp(-1j * Angle_end)

    return Y_MAX, Y_MID, Y_END

