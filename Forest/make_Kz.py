import math

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from osgeo import gdal
import matplotlib.pyplot as plt
from datetime import datetime

def make_kz(baseline,lambda_radar,range_line,angle_data,row,col):
    """
    这几个参数在envi
    例子：配准的结果 saocom_20240829_105058528_QS6_D_VV_slc_rsp_orb.sml 中有
    range_1 = 696371.072515945183113217353821
    range_2 = 703944.579486170201562345027924
    range_3 = 711514.339050670154392719268799
    box1 = (range_2 - range_1) / 4042
    box2 = (range_3 - range_2) / 4042
    lambda_radar = 0.235  # 雷达波长 (m)
    baseline = 107.514  # 垂直基线 (m)
    取一个平均值，就是距离向移动一个像元时，斜距变化大小
    :param base_line:
    :param lambda_radar:
    :param range:
    :param angle_file:
    :param row:
    :param col:
    :return:
    """
    print("-------开始计算KZ------")
    # 这里是计算斜距每移动一个距离向像元，变换多少，range是一个数组，数据可以在这几个参数在envi 配准的结果 saocom_20240829_105058528_QS6_D_VV_slc_rsp_orb.sml 中有
    range1 = (range_line[1] - range_line[0]) / (col / 2)
    range2 = (range_line[2] - range_line[1]) / (col / 2)
    range_ave = (range1 + range2) / 2
    Kz = np.zeros((row, col), dtype=np.float64)
    for i in range(row):
        for j in range(col):
            r = range_line[0] + j * range_ave
            Kz[i, j] = (4 * math.pi * baseline) / (lambda_radar * r * math.sin(angle_data[i,j]))
    print("-------kz计算结束------")
    return Kz


