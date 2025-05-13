import cmath

from osgeo import gdal
import numpy as np

"""
将复数数据的幅值 (magnitude) 存储为 ENVI 格式的单波段 float 影像。

参数：
- output_path: 输出文件路径 (如 "filtered_image_830_float.dat")
- complex_array: 复数数据 (numpy 数组)
- reference_file: 参考 ENVI 文件 (用于获取地理坐标信息)
"""
def save_float_as_envi(output_path, complex_array, reference_file):

    ref_dataset = gdal.Open(reference_file, gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName('ENVI')

    rows, cols = complex_array.shape

    # 计算幅值（转换为 float）
    # 创建输出文件
    out_dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)  # 1 个 float 波段

    # 复制地理坐标信息
    out_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_dataset.SetProjection(ref_dataset.GetProjection())

    # 写入 float 数据
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(complex_array)

    # 关闭数据集
    out_band.FlushCache()
    print(f"相位数据已保存为 ENVI 格式 float 文件: {output_path}")
def save_array_as_tiff(output_path, data_array, reference_file):
    """
    将输入的二维数组保存为 TIFF 格式的单波段影像

    参数：
    - output_path: 输出文件路径 (如 "output_image.tif")
    - data_array: 输入的 numpy 数组 (二维，支持 float32/int16 等类型)
    - reference_file: 参考文件路径 (用于获取地理坐标信息)
    """
    # 打开参考文件获取地理坐标信息
    ref_dataset = gdal.Open(reference_file, gdal.GA_ReadOnly)

    # 创建 GTiff 驱动
    driver = gdal.GetDriverByName('ENVI')

    # 获取数组维度
    rows, cols = data_array.shape

    # 自动推断 GDAL 数据类型
    dtype_map = {
        np.float32: gdal.GDT_Float32,
        np.int16: gdal.GDT_Int16,
        np.uint16: gdal.GDT_UInt16
    }
    gdal_dtype = dtype_map.get(data_array.dtype, gdal.GDT_Float32)

    # 创建输出数据集
    out_dataset = driver.Create(
        output_path,
        xsize=cols,
        ysize=rows,
        bands=1,
        eType=gdal_dtype
    )

    # 复制地理坐标信息
    out_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_dataset.SetProjection(ref_dataset.GetProjection())

    # 写入数据
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(data_array)

    # 确保数据写入磁盘
    out_band.FlushCache()
    print(f"成功保存 文件: {output_path}")