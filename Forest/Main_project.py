
from osgeo import gdal
from datetime import datetime
import make_pauli
import make_best
import numpy as np
import input_data
import make_Kz
import math
import make_ground
import Rvog
import Save_Height_envi
import ground_remove
import h5py
"""
    data_ : 是指的卫星数据的日期，例如20240829；
    number_id 指的是卫星标号：例如105058528
    envi配准后的结果标识：
    saocom_20240829_105058528_QS6_D_HH_slc_rsp

"""
# ———————————— 这里是读取数据的函数————————————————
# # 构建路径文件
# file_head = "saocom_"
# file_mid = "_QS6_D_"
# file_end = "_slc_rsp"
# Polarization = ["HH","HV","VH","VV"]
# lambda_radar = 0.235  # 雷达波长 (m)
# baseline = 107.514  # 垂直基线 (m)
#
# # start_time = datetime.now()
# # print("程序启动时间为：",start_time)
# # #
# #开始读取文件  注意，这里的 HV 是 HV 和 VH 的平均值
# array_HH_830, array_VV_830, array_HV_830, array_VH_830 = input_data.read_data_from_file(
#     r"E:\forest\Project\Forest\830_4042", '830', Polarization)
#
# array_HH_909, array_VV_909, array_HV_909, array_VH_909 = input_data.read_data_from_file(
#     r"E:\forest\Project\Forest\909_4042", '909', Polarization)
#
# #获取影像行列数
# row,col = array_VH_909.shape
# print(row,col)
# """
# #     md这里的所有计算都是像素为单位的，避免出错
# # """
# # 计算 Pauli 分解
# print("-------- 开始计算k1与k1_T(k1的共轭转置)----------")
# array_830_k1, array_830_k1_T, = make_pauli.make_pauli(
#     array_HH_830,array_VV_830,array_HV_830,
#     "E:/forest/Project/Forest/830_4042/array_830_k1.h5",
#     "E:/forest/Project/Forest/830_4042/array_830_k1_T.h5",
#     row,col,
#     'array_830_k1',
#     'array_830_k1_T')
# print("--------主图像k1计算结束，开始计算辅图像k2----------")
# array_909_k2, array_909_k2_T = make_pauli.make_pauli(
#     array_HH_909, array_VV_909, array_HV_909,
#     "E:/forest/Project/Forest/909_4042/array_909_k2.h5",
#     "E:/forest/Project/Forest/909_4042/array_909_k2_T.h5",
#     row,col,
#     'array_909_k2',
#     'array_909_k2_T'
# )
# print("-------- 辅图像k2计算结束 ----------")
# #读取保存的h5结果
# cow,rol = 24190,4042
# with h5py.File("830_4042/array_830_k1.h5",'r',) as f:
#     array_830_k1 = f['array_830_k1'][:]
#     print(array_830_k1.shape)
#
# with h5py.File("830_4042/array_830_k1_T.h5",'r') as f:
#     array_830_k1_T = f['array_830_k1_T'][:]
#     print(array_830_k1_T.shape)
#
# T11 = array_830_k1 @ array_830_k1_T
# with h5py.File("E:/forest/Project/Forest/T11.h5",'w') as f:
#     T11 = f.create_dataset("T11",data=T11,dtype=T11.dtype)
#     print("T11计算完成，并保存结果")
#
# with h5py.File("909_4042/array_909_k2.h5", 'r') as f:
#     array_909_k2 = f['array_909_k2'][:]
#     print("array_909_k2.shape", array_909_k2.shape)
#
# with h5py.File("909_4042/array_909_k2_T.h5", 'r') as f:
#     array_909_k2_T = f['array_909_k2_T'][:]
#     print("array_909_k2_T.shape", array_909_k2_T.shape)
#
# T22 = array_909_k2 @ array_909_k2_T
#
# with h5py.File("E:/forest/Project/Forest/T22.h5",'w') as f:
#     T22 = f.create_dataset("T22",data=T22,dtype=T22.dtype)
#     print("T22.dtype", T22.dtype)
#     print("T22计算完成，并保存结果")

with h5py.File("830_4042/array_830_k1.h5", 'r') as f:
    array_830_k1 = f['array_830_k1'][:]
    print("array_830_k1.shape", array_830_k1.shape)

with h5py.File("909_4042/array_909_k2_T.h5", 'r') as f:
    array_909_k2_T = f['array_909_k2_T'][:]
    print("array_909_k2_T.shape", array_909_k2_T.shape)

Omage12 = array_830_k1 @ array_909_k2_T

#TODO 这里我电脑计算内存不够了，明天上午使用工位电脑进行计算尝试一下：

master_orb_file = r'E:\forest\Project\830\saocom_20240829_105058528_QS6_D_HH_slc_rsp_orb.sml'
slave_orb_file = r'E:\forest\Project\909\saocom_20240906_105435075_QS6_D_HH_slc_rsp_orb.sml'

data_lat = gdal.Open(r'E:\forest\Project\Forest\830_4042\lat.dat',gdal.GA_ReadOnly)
lat = data_lat.GetRasterBand(1).ReadAsArray()
data_lon = gdal.Open(r'E:\forest\Project\Forest\830_4042\lon.dat',gdal.GA_ReadOnly)
lon = data_lon.GetRasterBand(1).ReadAsArray()

Omage12 = ground_remove.remove_ground(Omage12,lon,lat,master_orb_file,slave_orb_file)
with h5py.File("E:/forest/Project/Forest/Omage12.h5",'w') as f:
    Omage12 = f.create_dataset("Omage12",data=Omage12,dtype=Omage12.dtype)
    print("Omage12计算完成，并去除平地相位，保存结果")

"""
    创建新矩阵,计算T6、以及复相干优化
"""

# with h5py.File("T11.h5", 'r') as f:
#     T11 = f['T11'][:]
#     print(T11.shape)
#     print("T11成功读取")
# # with h5py.File("T22.h5", 'r') as f:
# #     T22 = f['T22'][:]
# #     print(T22.shape)
# #     print("T22成功读取")
# with h5py.File("Omage12.h5", 'r') as f:
#     Omage12 = f['Omage12'][:]
#     print(Omage12.shape)
#     print("Omage12成功读取")
# print("-------- T6、复相干优化----------")
# # Y_MAX, Y_MID, Y_END = make_best.make_T6_and_MAX(T11, T22, Omage12, row,col)
# Ymax,Ymin = make_best.pdopt(T11,Omage12,numph=30, step=100, reg=0.0, returnall=False)
# print("---------T6、复相干优化计算结束----------")
#
# """
#     这里要提前获取一下斜距、以及los中获取角度数据，转为弧度
#     这几个参数在envi 配准的结果
#     例子：saocom_20240829_105058528_QS6_D_VV_slc_rsp_orb.sml 中有
#     range_1 = 696371.072515945183113217353821
#     range_2 = 703944.579486170201562345027924
#     range_3 = 711514.339050670154392719268799
# """
# range = [696371.072515945183113217353821, 703944.579486170201562345027924,711514.339050670154392719268799]
# print(range[1])
# angle_file = "E://forest//Project//angle_mask_resize.dat"
#
# data_ = gdal.Open(angle_file,gdal.GA_ReadOnly)
# data_angle = data_.GetRasterBand(1).ReadAsArray()
# print(data_angle.shape)
# data_angle = np.deg2rad(data_angle)
#
# kz = make_Kz.make_kz(baseline,lambda_radar,range,data_angle,row,col)
# with h5py.File("kz.h5","w") as f:
#     f.create_dataset("kz",data=kz,dtype=kz.dtype)
#     print("kz计算结束")
#
# print("-----------开始计算地表相位-----------")
# gama_ground = np.zeros((2,row,col),type = complex)
# gama_ground[0,:,:] = Y_MAX
# gama_ground[1,:,:] = Y_END
# ground = make_ground.make_ground(gama_ground,kz)
# print("-------- ground计算结束 --------")
#
# print("-----------开始计算高度结果-------")
#
# height, converged = Rvog.Rvog(Y_MAX, ground, data_angle, kz)
#
# print("---------高度结果计算结束，开始保存结果--------")
#
# Out_file = "Forest//height.dat"
# Save_Height_envi.save_float_as_envi(Out_file,height,angle_file)
# print("--------结束所有程序，完成高度计算---------")
# end_time = datetime.now()
# print("程序启动结束为：",end_time)
# print("程序总计用时为：",end_time - start_time)

