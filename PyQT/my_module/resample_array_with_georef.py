from osgeo import gdal, gdal_array  # 需要安装: pip install GDAL
import numpy as np

def resample_by_pixel_size(input_array, input_geotransform, input_projection, target_pixel_size_x, target_pixel_size_y):
    """
    根据目标像素大小对数组进行重采样，保持地理编码和数据类型

    参数:
        input_array: 原始数组 (2D 或 3D [bands, height, width])
        input_geotransform: 原始地理变换
        input_projection: 投影信息
        target_pixel_size_x: 目标像素大小（X方向，以米或度为单位）
        target_pixel_size_y: 目标像素大小（Y方向，以米或度为单位）

    返回:
        tuple: (重采样后的数组, 新的地理变换, 投影)
    """
    if input_array.ndim == 2:
        input_array = np.expand_dims(input_array, axis=0)

    num_bands, orig_height, orig_width = input_array.shape
    dtype = input_array.dtype

    # 原始像素大小
    orig_pixel_size_x = input_geotransform[1]
    orig_pixel_size_y = abs(input_geotransform[5])  # 注意Y方向通常为负

    # 新的尺寸
    new_width = int((orig_width * orig_pixel_size_x) / target_pixel_size_x)
    new_height = int((orig_height * orig_pixel_size_y) / target_pixel_size_y)

    # 构造新的地理变换（左上角坐标不变）
    new_geotransform = (
        input_geotransform[0],
        target_pixel_size_x,
        input_geotransform[2],
        input_geotransform[3],
        input_geotransform[4],
        -target_pixel_size_y
    )

    mem_driver = gdal.GetDriverByName('MEM')
    src_ds = mem_driver.Create('', orig_width, orig_height, num_bands, gdal_array.NumericTypeCodeToGDALTypeCode(dtype))
    src_ds.SetGeoTransform(input_geotransform)
    src_ds.SetProjection(input_projection)

    for i in range(num_bands):
        src_ds.GetRasterBand(i + 1).WriteArray(input_array[i])

    dst_ds = mem_driver.Create('', new_width, new_height, num_bands, gdal_array.NumericTypeCodeToGDALTypeCode(dtype))
    dst_ds.SetGeoTransform(new_geotransform)
    dst_ds.SetProjection(input_projection)

    gdal.ReprojectImage(src_ds, dst_ds, None, None, gdal.GRA_NearestNeighbour)

    resampled_array = np.zeros((num_bands, new_height, new_width), dtype=dtype)
    for i in range(num_bands):
        resampled_array[i] = dst_ds.GetRasterBand(i + 1).ReadAsArray()

    if resampled_array.shape[0] == 1:
        resampled_array = resampled_array[0]

    return resampled_array, new_geotransform, input_projection

def resample_by_target_shape(input_array, input_geotransform, input_projection, target_rows, target_cols):
    """
    根据目标行列数对数组进行重采样（保持地理范围，改变像元大小）

    参数:
        input_array: 原始数组 (2D 或 3D: [bands, height, width])
        input_geotransform: 原始地理变换 (6 元组)
        input_projection: 投影信息
        target_rows: 目标图像的行数（高度）
        target_cols: 目标图像的列数（宽度）

    返回:
        tuple: (重采样后的数组, 新的地理变换, 投影)
    """
    if input_array.ndim == 2:
        input_array = np.expand_dims(input_array, axis=0)

    num_bands, orig_height, orig_width = input_array.shape
    dtype = input_array.dtype

    # 计算原图的空间范围
    origin_x = input_geotransform[0]
    pixel_width = input_geotransform[1]
    rotation_x = input_geotransform[2]
    origin_y = input_geotransform[3]
    rotation_y = input_geotransform[4]
    pixel_height = input_geotransform[5]

    total_width = orig_width * pixel_width
    total_height = orig_height * abs(pixel_height)

    # 新像元大小
    new_pixel_width = total_width / target_cols
    new_pixel_height = total_height / target_rows

    # 构造新的地理变换（保持左上角坐标不变）
    new_geotransform = (
        origin_x,
        new_pixel_width,
        rotation_x,
        origin_y,
        rotation_y,
        -new_pixel_height
    )

    # 构建 GDAL 内存数据集并写入数据
    mem_driver = gdal.GetDriverByName('MEM')
    src_ds = mem_driver.Create('', orig_width, orig_height, num_bands, gdal_array.NumericTypeCodeToGDALTypeCode(dtype))
    src_ds.SetGeoTransform(input_geotransform)
    src_ds.SetProjection(input_projection)
    for i in range(num_bands):
        src_ds.GetRasterBand(i + 1).WriteArray(input_array[i])

    # 创建目标内存数据集
    dst_ds = mem_driver.Create('', target_cols, target_rows, num_bands, gdal_array.NumericTypeCodeToGDALTypeCode(dtype))
    dst_ds.SetGeoTransform(new_geotransform)
    dst_ds.SetProjection(input_projection)

    # 进行重采样（默认最近邻）
    gdal.ReprojectImage(src_ds, dst_ds, None, None, gdal.GRA_NearestNeighbour)

    # 读取输出结果
    resampled_array = np.zeros((num_bands, target_rows, target_cols), dtype=dtype)
    for i in range(num_bands):
        resampled_array[i] = dst_ds.GetRasterBand(i + 1).ReadAsArray()

    if resampled_array.shape[0] == 1:
        resampled_array = resampled_array[0]

    return resampled_array, new_geotransform, input_projection