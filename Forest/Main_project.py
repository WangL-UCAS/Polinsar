
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
import h5py
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
lambda_radar = 0.235  # 雷达波长 (m)
baseline = 107.514  # 垂直基线 (m)

start_time = datetime.now()
print("程序启动时间为：",start_time)

#开始读取文件  注意，这里的 HV 是 HV 和 VH 的平均值
array_HH_830, array_VV_830, array_HV_830, array_VH_830 = input_data.read_data_from_file(
    "830", file_head, file_mid, file_end, Polarization, 20240829, 105058528)

array_HH_909, array_VV_909, array_HV_909, array_VH_909 = input_data.read_data_from_file(
    "909", file_head, file_mid, file_end, Polarization, 20240906, 105435075)

#获取影像行列数
row,col = array_HH_830.shape
"""
    md这里的所有计算都是像素为单位的，避免出错
"""
# 计算 Pauli 分解
print("-------- 开始计算k1与k1_T(k1的共轭转置)----------")
array_830_k1, array_830_k1_T = make_pauli.make_pauli(array_HH_830,array_VV_830,array_HV_830,row,col)
print("--------主图像k1计算结束，开始计算辅图像k2----------")
array_909_k2,array_909_k2_T = make_pauli.make_pauli(array_HH_909, array_VV_909, array_HV_909,row,col)
print("-------- 辅图像k2计算结束 ----------")

T11 = array_830_k1 @ array_830_k1_T
T22 = array_909_k2 @ array_909_k2_T
Omage12 = array_830_k1 @ array_909_k2_T
"""
    创建新矩阵,计算T6、以及复相干优化
"""
print("-------- T6、复相干优化----------")
Y_MAX, Y_MID, Y_END = make_best.make_T6_and_MAX(T11, T22, Omage12, row,col)
print("---------T6、复相干优化计算结束----------")

"""
    这里要提前获取一下斜距、以及los中获取角度数据，转为弧度
    这几个参数在envi 配准的结果 
    例子：saocom_20240829_105058528_QS6_D_VV_slc_rsp_orb.sml 中有
    range_1 = 696371.072515945183113217353821
    range_2 = 703944.579486170201562345027924
    range_3 = 711514.339050670154392719268799
"""
range = np.array([696371.072515945183113217353821, 703944.579486170201562345027924,711514.339050670154392719268799])
angle_file = "E://forest//Project//angle_mask_resize.dat"

data_angle = gdal.Open(angle_file,gdal.GA_ReadOnly).GetRasterBand(1).ReadAsArray()
data_angle = math.radians(data_angle)

kz = make_Kz.make_kz(baseline,lambda_radar,range,data_angle,row,col)

print("-----------开始计算地表相位-----------")
gama_ground = np.zeros((2,row,col),type = complex)
gama_ground[0,:,:] = Y_MAX
gama_ground[1,:,:] = Y_END
ground = make_ground.make_ground(gama_ground,kz)
print("-------- ground计算结束 --------")

print("-----------开始计算高度结果-------")

height, converged = Rvog.Rvog(Y_MAX, ground, data_angle, kz)

print("---------高度结果计算结束，开始保存结果--------")

Out_file = "Forest//height.dat"
Save_Height_envi.save_float_as_envi(Out_file,height,angle_file)
print("--------结束所有程序，完成高度计算---------")
end_time = datetime.now()
print("程序启动结束为：",end_time)
print("程序总计用时为：",end_time - start_time)



