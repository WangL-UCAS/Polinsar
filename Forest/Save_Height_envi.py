from osgeo import gdal
import numpy as np


def save_float_as_envi(output_path, complex_array, reference_file):
    """
    将复数数据的幅值 (magnitude) 存储为 ENVI 格式的单波段 float 影像。

    参数：
    - output_path: 输出文件路径 (如 "filtered_image_830_float.dat")
    - complex_array: 复数数据 (numpy 数组)
    - reference_file: 参考 ENVI 文件 (用于获取地理坐标信息)
    """
    ref_dataset = gdal.Open(reference_file, gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName('ENVI')

    rows, cols = complex_array.shape

    # 计算幅值（转换为 float）
    float_array = np.abs(complex_array).astype(np.float32)

    # 创建输出文件
    out_dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)  # 1 个 float 波段

    # 复制地理坐标信息
    out_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_dataset.SetProjection(ref_dataset.GetProjection())

    # 写入 float 数据
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(float_array)

    # 关闭数据集
    out_band.FlushCache()
    out_dataset = None
    print(f"幅值数据已保存为 ENVI 格式 float 文件: {output_path}")