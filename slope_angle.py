import os

import h5py
import numba
import numpy as np
from osgeo import gdal, gdal_array

def compute_and_save_slope(dem_file, output_path, pixel_size_x, pixel_size_y):
    """
    计算坡度信息并保存为 ENVI 格式。

    :param dem_file: 输入的高程数据文件路径
    :param output_path: 输出坡度数据的文件路径
    :param pixel_size_x: X 方向的像素间隔
    :param pixel_size_y: Y 方向的像素间隔
    """
    # 打开数据集
    dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"无法打开文件: {dem_file}")

    # 读取高程数据
    band = dataset.GetRasterBand(1)
    elevation = band.ReadAsArray()

    # 计算 X（东西）和 Y（南北）方向的梯度
    dz_dx, dz_dy = np.gradient(elevation, pixel_size_x, pixel_size_y)

    # 计算坡度（以度为单位）
    slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)) * (180 / np.pi)

    # 保存坡度数据为 ENVI 格式
    save_slope_envi(output_path, slope, dem_file)

    # 关闭数据集
    dataset = None


def save_slope_envi(output_path, slope_data, reference_file):
    """
    以 ENVI 格式保存坡度数据。

    :param output_path: 输出文件路径
    :param slope_data: 计算得到的坡度数据
    :param reference_file: 参考的地理信息文件
    """
    ref_dataset = gdal.Open(reference_file, gdal.GA_ReadOnly)
    if ref_dataset is None:
        raise FileNotFoundError(f"无法打开参考文件: {reference_file}")

    driver = gdal.GetDriverByName('ENVI')
    rows, cols = slope_data.shape
    out_dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)

    out_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_dataset.SetProjection(ref_dataset.GetProjection())

    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(slope_data)
    out_band.FlushCache()

    out_band = None
    out_dataset = None
    ref_dataset = None
    print(f"坡度文件已保存: {output_path}")


# 示例使用

#dem_file z.rdr文件位置, output_path 输出位置, pixel_size_x 空间分辨率, pixel_size_y空间分辨率 （这里不确定）
compute_and_save_slope(r"E:\forest\Project\Forest\HZZ-2\HZZ-2-830-909\data\z.dat", r"E:\forest\Project\Forest\HZZ-2\HZZ-2-830-909\data\slope.dat",  7.8182794096764327562709695485, 3.99259963345670998435821275052)
#compute_and_save_slope("909/z.rdr.full","909/slope_angle.dat",7.742726833957243, 3.99279930026319)




