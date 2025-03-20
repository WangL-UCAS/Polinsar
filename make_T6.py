import numpy as np
from osgeo import gdal
import math


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
array_HH_909 = (array_HV_909 + array_VH_909) / 2

print("测试一下数据读取是否成功读取：", array_HH_909[5000][2000],"----" ,array_VV_909[5000][2000])

"""
    make_pauli 是创建pauli散射的 K 矩阵
"""
def make_pauli(array_HH, array_VV, array_HV):
    k1 = (array_HH + array_VV) / np.sqrt(2)
    k2 = (array_HH - array_VV) / np.sqrt(2)
    k3 = 2 * array_HV / np.sqrt(2)
    return k1, k2, k3

# 计算 Pauli 分解
array_830_k1, array_830_k2, array_830_k3 = make_pauli(array_HH_830, array_VV_830, array_HV_830)
array_909_k1, array_909_k2, array_909_k3 = make_pauli(array_HH_909, array_VV_909, array_HV_909)

print("-------pauli 计算结束-------")

"""
    ------- 计算 T6 并保存---------
    transpose_conjugate = matrix.T.conj()  矩阵转置共轭函数 matrix 是原始函数
     k1 = np.array([array_mask_k1, array_mask_k2, array_mask_k3]).T

"""
def make_T6(array_mask_k1, array_mask_k2, array_mask_k3, array_slave_k1, array_slave_k2, array_slave_k3):

    """
    计算矩阵转置共轭
    参数说明： K1是k6矩阵中的前三项、k2为后三项 ;K6 = [array_830_k1, array_830_k2, array_830_k3, array_909_k1, array_909_k2, array_909_k3]
    k1_T 表示是k1 的转置，也就是从三行一列转为一行三列用于后续的计算
    而其中元素（array_830_k1、、、、）的的转置共轭在外面已经计算  例如： array_mask_k1_T = array_mask_k1.T.conj()
    Omaga_11 表示T6矩阵中的极化干涉矩阵，如果将T6的 6x6矩阵分为四块；那么Omaga_11 表示的是右上角的 3x3矩阵

    """

    array_mask_k1_T = array_mask_k1.T.conj()
    array_mask_k2_T = array_mask_k2.T.conj()
    array_mask_k3_T = array_mask_k3.T.conj()
    k1 = np.array([[array_mask_k1],[array_mask_k2],[array_mask_k3]]) # 这里的数组大小为 (3,1,row,col)
    k1_T = np.array([[array_mask_k1_T,array_mask_k2_T,array_mask_k3_T]]) # 这里由于上面对元素进行了转置并在外面手动加入转置因此这里的数组大小为(1,3,col,row)
    T11 = k1 * k1_T

    array_slave_k1_T = array_slave_k1.T.conj()
    array_slave_k2_T = array_slave_k2.T.conj()
    array_slave_k3_T = array_slave_k3.T.conj()

    k2 = np.array([[array_slave_k1],[array_slave_k2],[array_slave_k3]])
    k2_T = np.array([[array_slave_k1_T, array_slave_k2_T, array_slave_k3_T]])
    T22 = k2 * k2_T

    Omaga_11 = k2 * k1_T
    return T11, T22, Omaga_11