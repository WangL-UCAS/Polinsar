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
    k1 = np.array([[array_mask_k1,array_mask_k2,array_mask_k3]])
    k1_T = np.array([[array_mask_k1_T,array_mask_k2_T,array_mask_k3_T]])
    T11 = np.zeros((3, 3))

    ## 这里采用循环计算矩阵相乘，得到最后的T11
    for i in range(3):
        for j in range(3):
            T11[i][j] = np.dot(k1[i],k1_T[j])

    array_slave_k1_T = array_slave_k1.T.conj()
    array_slave_k2_T = array_slave_k2.T.conj()
    array_slave_k3_T = array_slave_k3.T.conj()

    k2 = np.array([[array_slave_k1,array_slave_k2,array_slave_k3]])
    k2_T = np.array([[array_slave_k1_T, array_slave_k2_T, array_slave_k3_T]])

    T22 = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            T22[i][j] = np.dot(k2[i],k2_T[i])

    Omaga_11 = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Omaga_11[i][j] = np.dot(k2[i],k1_T[j])
    Omaga_11 = k2 * k1_T
    return T11, T22, Omaga_11

T11, T22, Omaga_11 = make_T6(array_830_k1, array_830_k2, array_830_k3, array_909_k1, array_909_k2, array_909_k3)
"""
    
    计算相干性优化？ copy的kapok代码，调试使
    
"""
# def pdopt(tm, om, numph=30, step=50, reg=0.0, returnall=False):
#     # 获取输入矩阵tm的维度（azimuth, range, n, n），n为极化通道数
#     dim = np.shape(tm)
#
#     # 矩阵正则化（防止奇异矩阵）
#     if reg > 0:
#         # 创建与tm同维度的正则化矩阵
#         regmat = np.zeros(dim, dtype='complex64')
#         # 每个位置填充单位矩阵乘以正则化系数和tm的迹
#         regmat[:, :] = np.eye(dim[2])
#         regmat = regmat * reg * np.trace(tm, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis]
#         tm = tm + regmat  # 添加到原矩阵
#
#         # 对om矩阵执行相同正则化
#         regmat = np.zeros(dim, dtype='complex64')
#         regmat[:, :] = np.eye(dim[2])
#         regmat = regmat * reg * np.trace(om, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis]
#         om = om + regmat
#         del regmat  # 释放内存
#
#     # 初始化存储结果的数组
#     cohsize = (dim[0], dim[1])  # 方位和距离向像素数
#     cohdiff = np.zeros(cohsize, dtype='float32')  # 最大相干差异
#     gammamax = np.zeros(cohsize, dtype='complex64')  # 最大相干值
#     gammamin = np.zeros(cohsize, dtype='complex64')  # 最小相干值
#
#     # 如果需返回全部结果，初始化最小差异相关数组
#     if returnall:
#         mincohdiff = np.ones(cohsize, dtype='float32') * 99
#         gammaminormax = np.zeros(cohsize, dtype='complex64')
#         gammaminormin = np.zeros(cohsize, dtype='complex64')
#
#     # 初始化极化权重向量存储数组（若需要返回）
#     weightsize = (dim[0], dim[1], dim[3])
#     wmax = np.zeros(weightsize, dtype='complex64')  # 最大相干对应权重
#     wmin = np.zeros(weightsize, dtype='complex64')  # 最小相干对应权重
#
#     # 主循环：遍历不同相位旋转角度
#     for Ph in np.arange(0, numph):
#         Pr = Ph * np.pi / numph  # 计算当前相位弧度值
#
#         # 打印进度（覆盖式输出）
#         print('kapok.cohopt.pdopt | Current Progress: ' + str(
#             np.round(Pr / np.pi * 100, decimals=2)) + '%. (' + time.ctime() + ')     ', end='\r')
#
#         # 分块处理（提升大矩阵运算效率）
#         for az in range(0, dim[0], step):
#             azend = min(az + step, dim[0])  # 计算当前块结束位置
#
#             for rng in range(0, dim[1], step):
#                 rngend = min(rng + step, dim[1])
#
#                 # 提取当前数据块
#                 omblock = om[az:azend, rng:rngend]
#                 tmblock = tm[az:azend, rng:rngend]
#                 z12 = omblock.copy()
#
#                 # 相位旋转：给omega矩阵施加相位偏移
#                 z12 *= np.exp(1j * Pr)
#                 # 强制成为共轭对称矩阵（保证特征值为实数）
#                 z12 = 0.5 * (z12 + np.rollaxis(np.conj(z12), 3, 2))
#
#                 # 检查奇异矩阵（行列式为0）
#                 det = linalg.det(tmblock)
#                 ind = (det == 0)
#                 if np.any(ind):
#                     tmblock[ind] = np.eye(dim[3])  # 替换为单阵避免计算错误
#
#                 # 求解广义特征值问题：inv(T) * Ω 的特征值
#                 nu, w = linalg.eig(np.einsum('...ij,...jk->...ik',
#                                              linalg.inv(tmblock), z12))
#
#                 # 计算伴随矩阵（共轭转置）
#                 wH = np.rollaxis(np.conj(w), 3, 2)
#
#                 # 计算相干性分子项：w^H * Ω * w
#                 Tmp = np.einsum('...ij,...jk->...ik', omblock, w)
#                 Tmp12 = np.einsum('...ij,...jk->...ik', wH, Tmp)
#
#                 # 计算相干性分母项：w^H * T * w
#                 Tmp = np.einsum('...ij,...jk->...ik', tmblock, w)
#                 Tmp11 = np.einsum('...ij,...jk->...ik', wH, Tmp)
#
#                 # 创建索引网格用于后续取值
#                 azind = np.tile(np.arange(0, w.shape[0]), (w.shape[1], 1)).T
#                 rngind = np.tile(np.arange(0, w.shape[1]), (w.shape[0], 1))
#
#                 # 找到最小/最大特征值对应的索引
#                 lmin = np.argmin(nu, axis=2)
#                 gmin = Tmp12[azind, rngind, lmin, lmin] / np.abs(Tmp11[azind, rngind, lmin, lmin])
#
#                 lmax = np.argmax(nu, axis=2)
#                 gmax = Tmp12[azind, rngind, lmax, lmax] / np.abs(Tmp11[azind, rngind, lmax, lmax])
#
#                 # 判断当前差异是否更大
#                 ind = (np.abs(gmax - gmin) > cohdiff[az:azend, rng:rngend])
#
#                 # 更新最大差异结果
#                 if np.any(ind):
#                     azupdate, rngupdate = np.where(ind)
#                     # 更新全局数组对应位置的值
#                     cohdiff[az + azupdate, rng + rngupdate] = np.abs(gmax - gmin)[azupdate, rngupdate]
#                     gammamax[az + azupdate, rng + rngupdate] = gmax[azupdate, rngupdate]
#                     gammamin[az + azupdate, rng + rngupdate] = gmin[azupdate, rngupdate]
#
#                     # 若需要返回权重向量
#                     if returnall:
#                         wmax[az + azupdate, rng + rngupdate, :] = w[azupdate, rngupdate, :, lmax[azupdate, rngupdate]]
#                         wmin[az + azupdate, rng + rngupdate, :] = w[azupdate, rngupdate, :, lmin[azupdate, rngupdate]]
#
#                 # 处理最小差异记录（若需要返回全部结果）
#                 if returnall:
#                     ind = (np.abs(gmax - gmin) < mincohdiff[az:azend, rng:rngend])
#                     if np.any(ind):
#                         azupdate, rngupdate = np.where(ind)
#                         mincohdiff[az + azupdate, rng + rngupdate] = np.abs(gmax - gmin)[azupdate, rngupdate]
#                         gammaminormax[az + azupdate, rng + rngupdate] = gmax[azupdate, rngupdate]
#                         gammaminormin[az + azupdate, rng + rngupdate] = gmin[azupdate, rngupdate]
#
#     # 完成提示
#     print('kapok.cohopt.pdopt | Optimization complete. (' + time.ctime() + ')          ')
#
#     # 根据标志返回不同结果
#     if returnall:
#         return (gammamax, gammamin,
#                 gammaminormax, gammaminormin,
#                 wmax, wmin)
#     else:
#         return gammamax, gammamin
#
# gama_max, gama_min = pdopt(T11, Omaga_11,numph=30, step=50, reg=0.0, returnall=False )
