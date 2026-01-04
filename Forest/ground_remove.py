import numpy as np
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
import math
import matplotlib.pyplot as plt
from osgeo import gdal
import mlutilooking
def remove_ground(sint_file, Omage,baseline):

    sint = gdal.Open(sint_file,gdal.GA_ReadOnly)
    sint = sint.GetRasterBand(1).ReadAsArray()
    sint = mlutilooking.multilook_float_numba(sint,2,4)
    """
        tmd。测试结果这里在垂直基线正负不影响，都是乘1j
        (-0.062769346+0.024656001j)  计算的去平
        (-0.062769346+0.024656001j)  ENVI 的去平结果
        
        int_data = gdal.Open(r'E:\out\FP-2-1005-1018_int', gdal.GA_ReadOnly)
        int_data = int_data.GetRasterBand(1).ReadAsArray()
        
        sint_data = gdal.Open(r'E:\out\SINT.dat', gdal.GA_ReadOnly)
        sint_data = sint_data.GetRasterBand(1).ReadAsArray()
        
        dint = gdal.Open(r'E:\out\FP-2-1005-1018_dint', gdal.GA_ReadOnly)
        dint = dint.GetRasterBand(1).ReadAsArray()
        
        # 计算相位
        sint = np.exp(1j * sint_data)
        DINT  = int_data * sint
        print(DINT[12000,2000])
        print(dint[12000,2000])
    """
    if baseline > 0:
        sint = np.exp(1j * sint)
    else:
        sint = np.exp(1j * sint)
    Omage12_flat = Omage * sint[:,:,np.newaxis,np.newaxis]

    return Omage12_flat




