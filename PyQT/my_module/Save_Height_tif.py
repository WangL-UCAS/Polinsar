
import numpy as np
from osgeo import gdal


def save_array_as_tiff(output_path, data_array, projection, geotransform):
    """
    将输入的二维数组保存为 TIFF 格式的单波段影像

    参数：
    - output_path: 输出文件路径（如 "output.tif"）
    - data_array: 2D numpy 数组
    - projection: 投影信息（WKT 字符串）
    - geotransform: 地理变换信息（六元组）
    """

    rows, cols = data_array.shape

    # 推断数据类型
    dtype_map = {
        np.dtype('float32'): gdal.GDT_Float32,
        np.dtype('int16'): gdal.GDT_Int16,
        np.dtype('uint16'): gdal.GDT_UInt16,
        np.dtype('uint8'): gdal.GDT_Byte
    }
    gdal_dtype = dtype_map.get(data_array.dtype, gdal.GDT_Float32)

    # 创建 GTiff 驱动（或 'ENVI' 视具体需求）
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_path, cols, rows, 1, gdal_dtype)

    # 设置地理信息
    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)

    # 写入波段数据
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(data_array)
    out_band.FlushCache()

    print(f"✅ 成功保存 TIFF 文件: {output_path}")

