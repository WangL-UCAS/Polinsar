import os
import sys
import numpy as np
from osgeo import gdal, osr
import pyresample as pr

def radar2ll_pr(outpath, datafile, data, lat, lon, outformat='ENVI',
                nodataval=0, tr=0.000135, **kwargs):
    """
    将雷达数据重投影到经纬度坐标系

    参数：
    outpath   : 输出目录路径
    datafile  : 输出文件名
    data      : 输入数据数组
    lat       : 输入纬度数组
    lon       : 输入经度数组
    outformat : 输出格式 (默认ENVI)
    nodataval : 无效值 (默认0)
    tr        : 输出分辨率 (度)，默认约15米分辨率

    返回：
    (输出数据, 投影WKT, 地理变换参数)
    """

    # 处理无效值
    if nodataval is None:
        nodataval = 0.0

    # 创建输出目录
    if outpath:
        outpath = os.path.abspath(outpath)
        os.makedirs(outpath, exist_ok=True)

    # 解析分辨率参数
    if isinstance(tr, tuple):
        dlon, dlat = map(abs, (float(tr[0]), float(tr[1])))
    else:
        dlon = dlat = abs(float(tr))

    # 计算数据范围
    lat_bounds = np.nanmax(lat), np.nanmin(lat)
    lon_bounds = np.nanmin(lon), np.nanmax(lon)

    # 计算网格参数（含边界缓冲）
    PAD = 10.5  # 边界缓冲系数
    ilat0 = int(lat_bounds[0] / dlat + PAD)
    ilat1 = int(lat_bounds[1] / dlat - PAD)
    ilon0 = int(lon_bounds[0] / dlon - PAD)
    ilon1 = int(lon_bounds[1] / dlon + PAD)

    nlat = ilat0 - ilat1
    nlon = ilon1 - ilon0

    # 计算实际输出范围
    lat0 = ilat0 * dlat
    lon0 = ilon0 * dlon

    # 地理变换参数计算 (左上角坐标)
    ullat = lat0 + 0.5 * dlat  # 纬度（北向）
    ullon = lon0 - 0.5 * dlon  # 经度（西向）

    # 创建空间参考系统
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84坐标系
    projection = srs.ExportToWkt()

    # 地理变换参数 (GDAL格式)
    geotransform = (
        ullon,  # 左上角经度
        dlon,   # 经度分辨率
        0,      # 旋转项（东西方向）
        ullat,  # 左上角纬度
        0,      # 旋转项（南北方向）
        -dlat   # 纬度分辨率（负值表示北向南增加）
    )

    # 生成目标网格定义
    x_coords = lon0 + dlon * np.arange(nlon)
    y_coords = lat0 - dlat * np.arange(nlat)
    X, Y = np.meshgrid(x_coords, y_coords)
    target_grid = pr.geometry.GridDefinition(lons=X, lats=Y)

    # 定义输入swath
    input_swath = pr.geometry.SwathDefinition(lons=lon, lats=lat)

    # 重采样参数计算
    earth_radius = 6378137.0  # WGS84椭球长半轴
    dx = earth_radius * np.radians(dlat)
    sigma = pr.utils.fwhm2sigma(dx)  # 高斯平滑参数

    # 数据类型转换
    data = data.astype('float32')

    # 执行重采样
    resampled = pr.kd_tree.resample_gauss(
        input_swath, data, target_grid,
        radius_of_influence=3 * dx,
        sigmas=sigma,
        neighbours=kwargs.get('nn', 50),
        segments=1,
        fill_value=nodataval
    )

    # 输出数据处理
    out_data = resampled.astype('f4')

    # 直接使用GDAL驱动创建目标文件
    driver = gdal.GetDriverByName(outformat)
    output_file = os.path.join(outpath, datafile)
    dataset = driver.Create(output_file, nlon, nlat, 1, gdal.GDT_Float32)

    dataset.SetProjection(projection)
    dataset.SetGeoTransform(geotransform)

    band = dataset.GetRasterBand(1)
    band.WriteArray(out_data)
    band.SetNoDataValue(nodataval)

    dataset.FlushCache()
    dataset = None

    return out_data, projection, geotransform
