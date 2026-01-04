from osgeo import gdal, osr
import numpy as np

gdal.UseExceptions()

lon_path = r"E:\forest\Project\Forest\HZZ-2\HZZ-2-909-917-mlutilooking\data\lon.dat"
lat_path = r"E:\forest\Project\Forest\HZZ-2\HZZ-2-909-917-mlutilooking\data\lat.dat"
slope_path = r"E:\forest\Project\DEM\找dem\坡向.tif"
out_path = r"E:\forest\Project\Forest\HZZ-2\HZZ-2-909-917-mlutilooking\data\aspect.tif"

# -----------------------------
# 打开数据
# -----------------------------
lon_ds = gdal.Open(lon_path)
lat_ds = gdal.Open(lat_path)
slope_ds = gdal.Open(slope_path)

if lon_ds is None:
    raise RuntimeError(f"无法打开 lon 文件：{lon_path}")
if lat_ds is None:
    raise RuntimeError(f"无法打开 lat 文件：{lat_path}")
if slope_ds is None:
    raise RuntimeError(f"无法打开 slope 文件：{slope_path}")

lon_band = lon_ds.GetRasterBand(1)
lat_band = lat_ds.GetRasterBand(1)
slope_band = slope_ds.GetRasterBand(1)

# -----------------------------
# 读取 lon/lat 数据
# -----------------------------
lon = lon_band.ReadAsArray().astype(np.float64)
lat = lat_band.ReadAsArray().astype(np.float64)

if lon.shape != lat.shape:
    raise RuntimeError(f"lon 和 lat 尺寸不一致：lon={lon.shape}, lat={lat.shape}")

# -----------------------------
# nodata 处理
# -----------------------------
lon_nodata = lon_band.GetNoDataValue()
lat_nodata = lat_band.GetNoDataValue()
slope_nodata = slope_band.GetNoDataValue()

valid = np.ones(lon.shape, dtype=bool)

if lon_nodata is not None:
    valid &= lon != lon_nodata
if lat_nodata is not None:
    valid &= lat != lat_nodata

valid &= np.isfinite(lon) & np.isfinite(lat)

# -----------------------------
# slope GeoTransform 逆变换（兼容不同 GDAL 版本）
# -----------------------------
gt = slope_ds.GetGeoTransform()

inv_ret = gdal.InvGeoTransform(gt)

# 兼容情况1：返回 (success, inv_gt)
if isinstance(inv_ret, tuple) and len(inv_ret) == 2 and isinstance(inv_ret[0], (bool, int)):
    success = bool(inv_ret[0])
    inv_gt = inv_ret[1]
    if not success:
        raise RuntimeError("无法求逆 GeoTransform（InvGeoTransform 返回失败）")

# 兼容情况2：直接返回 6 元素 tuple（你现在的情况）
elif isinstance(inv_ret, tuple) and len(inv_ret) == 6:
    inv_gt = inv_ret

# 兼容情况3：返回 None/False
else:
    raise RuntimeError(f"无法解析 InvGeoTransform 返回值：{inv_ret}")

# -----------------------------
# 坐标系：lon/lat 默认 EPSG:4326
# -----------------------------
src_srs = osr.SpatialReference()
src_srs.ImportFromEPSG(4326)

# slope 的坐标系
dst_wkt = slope_ds.GetProjection()
if dst_wkt is None or dst_wkt.strip() == "":
    raise RuntimeError("slope.tif 没有投影信息（Projection 为空），无法确定 slope 的坐标系")

dst_srs = osr.SpatialReference()
dst_srs.ImportFromWkt(dst_wkt)

need_transform = not src_srs.IsSame(dst_srs)

if need_transform:
    ct = osr.CoordinateTransformation(src_srs, dst_srs)

# -----------------------------
# 输出数组
# -----------------------------
out = np.full(lon.shape, np.nan, dtype=np.float32)

# -----------------------------
# 遍历有效点采样 slope
# -----------------------------
rows, cols = np.where(valid)
total = len(rows)
print(f"有效点数量：{total}")

for i, (r, c) in enumerate(zip(rows, cols), start=1):
    x = float(lon[r, c])
    y = float(lat[r, c])

    # 如果 slope 不是 EPSG:4326，先转坐标
    if need_transform:
        x, y, _ = ct.TransformPoint(x, y)

    # 地理坐标 -> slope 像素坐标
    px, py = gdal.ApplyGeoTransform(inv_gt, x, y)
    px = int(np.floor(px))
    py = int(np.floor(py))

    # 越界跳过
    if px < 0 or py < 0 or px >= slope_ds.RasterXSize or py >= slope_ds.RasterYSize:
        continue

    # 读取 slope 值（1x1 window）
    val = slope_band.ReadAsArray(px, py, 1, 1)[0, 0]

    # nodata -> nan
    if slope_nodata is not None and val == slope_nodata:
        continue

    out[r, c] = val

    # 可选：进度输出
    if i % 1000000 == 0:
        print(f"进度：{i}/{total}")

# -----------------------------
# 写出 tif（输出网格用 lon_ds 的网格信息）
# -----------------------------
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(
    out_path,
    lon_ds.RasterXSize,
    lon_ds.RasterYSize,
    1,
    gdal.GDT_Float32,
    options=["TILED=YES", "COMPRESS=LZW"]
)

if out_ds is None:
    raise RuntimeError(f"无法创建输出文件：{out_path}")

# lon.dat 可能没有空间信息（GeoTransform/Projection）
lon_gt = lon_ds.GetGeoTransform(can_return_null=True)
lon_proj = lon_ds.GetProjection()

if lon_gt is None:
    print("警告：lon.dat 没有 GeoTransform，输出 tif 将没有正确空间定位。")
else:
    out_ds.SetGeoTransform(lon_gt)

if lon_proj is None or lon_proj.strip() == "":
    print("警告：lon.dat 没有 Projection，输出 tif 将没有正确投影。")
else:
    out_ds.SetProjection(lon_proj)

out_band = out_ds.GetRasterBand(1)

# 把 NaN 统一替换成 nodata（更通用）
out_nodata = -9999.0
out_band.SetNoDataValue(out_nodata)

out_write = out.copy()
out_write[~np.isfinite(out_write)] = out_nodata

out_band.WriteArray(out_write)
out_band.FlushCache()
out_ds.FlushCache()

# 关闭
out_ds = None
lon_ds = None
lat_ds = None
slope_ds = None

print("完成：", out_path)
