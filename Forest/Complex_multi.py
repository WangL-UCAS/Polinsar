import os
import numba
import numpy as np
from osgeo import gdal

# 复数多视处理函数
@numba.njit(parallel=True)
def multilook_complex_numba(data, range_looks, azimuth_looks):
    H, W = data.shape
    H_out = H // azimuth_looks
    W_out = W // range_looks
    downsampled = np.zeros((H_out, W_out), dtype=np.complex128)

    for i in numba.prange(H_out):
        for j in range(W_out):
            row_start = i * azimuth_looks
            row_end = (i + 1) * azimuth_looks
            col_start = j * range_looks
            col_end = (j + 1) * range_looks

            block = data[row_start:row_end, col_start:col_end]
            downsampled[i, j] = np.mean(block)

    return downsampled

# 保存复数 ENVI 文件
def save_envi_complex(array, out_path, ref_ds=None):
    driver = gdal.GetDriverByName("ENVI")
    out_ds = driver.Create(out_path, array.shape[1], array.shape[0], 1, gdal.GDT_CFloat64)
    out_ds.GetRasterBand(1).WriteArray(array)

    # 可选：复制地理参考信息
    if ref_ds:
        out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
        out_ds.SetProjection(ref_ds.GetProjection())

    out_ds.FlushCache()
    out_ds = None

# 参数配置
base_dir = r"E:\forest\Project\Forest\FP-1\FP-1-903-911\903"
polarizations = ['HH', 'VV', 'VH', 'HV']
azimuth_looks = 4
range_looks = 2

# 处理每个极化数据
for pol in polarizations:
    file_name = f"909-{pol}"
    input_path = os.path.join(base_dir, file_name)
    output_path = os.path.join(base_dir, f"{file_name}_ml.dat")

    if not os.path.exists(input_path):
        print(f"⚠️ 文件不存在：{input_path}，跳过。")
        continue

    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"❌ 无法读取：{input_path}")
        continue

    # 读取为复数数据
    arr = ds.GetRasterBand(1).ReadAsArray()
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128)
    else:
        arr = arr.astype(np.complex128)

    # 多视处理
    looked = multilook_complex_numba(arr, range_looks, azimuth_looks)

    # 保存结果
    save_envi_complex(looked, output_path, ref_ds=ds)
    print(f"✅ 已完成多视处理并保存：{output_path}")
