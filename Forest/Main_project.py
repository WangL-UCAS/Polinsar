
from osgeo import gdal
from datetime import datetime
import make_pauli
import make_best
import numpy as np
import input_data

"""
    data_ : 是指的卫星数据的日期，例如20240829；
    number_id 指的是卫星标号：例如105058528
    envi配准后的结果标识：
    saocom_20240829_105058528_QS6_D_HH_slc_rsp

"""
## ———————————— 这里是读取数据的函数————————————————
#构建路径文件
file_head = "saocom_"
file_mid = "_QS6_D_"
file_end = "_slc_rsp"
Polarization = ["HH","HV","VH","VV"]

start_time = datetime.now()
print("程序启动时间为：",start_time)

#开始读取文件  注意，这里的 HV 是 HV 和 VH 的平均值
array_HH_830, array_VV_830, array_HV_830, array_VH_830 = input_data.read_data_from_file("830", file_head, file_mid, file_end, Polarization, 20240829, 105058528)

array_HH_909, array_VV_909, array_HV_909, array_VH_909 = input_data.read_data_from_file("909", file_head, file_mid, file_end, Polarization, 20240906, 105435075)

#获取影像行列数
row,col = array_HH_830.shape
"""
    make_pauli 是创建pauli散射的 K 矩阵,
    md这里的计算都是像素为单位的，避免出错
"""
# 计算 Pauli 分解
print("-------- 开始计算pauli分解----------")
array_830_k1 = make_pauli.make_pauli(array_HH_830,array_VV_830,array_HV_830,row,col)
array_909_k2 = make_pauli.make_pauli(array_HH_909, array_VV_909, array_HV_909,row,col)

#计算共轭转置
array_830_k1_T = make_pauli.make_pauli(array_HH_830, array_VV_830, array_HV_830,row,col)
array_909_k2_T = make_pauli.make_pauli(array_HH_909, array_VV_909, array_HV_909,row,col)

"""
    创建新矩阵,计算T6、以及复相干优化
"""

print("-------- T6、复相干优化----------")
T11, T22, Omaga12, Y_MAX, Y_MID, Y_END = make_best.make_T6_and_MAX(array_830_k1,array_830_k1_T,array_909_k2,array_909_k2_T,row,col)

end_time = datetime.now()
print("程序启动结束为：",end_time)
print("程序总计用时为：",end_time - start_time)



