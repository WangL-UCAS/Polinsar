
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from osgeo import gdal
import matplotlib.pyplot as plt
from datetime import datetime


strat_time = datetime.now()
print("当前时间为：",datetime.now())
# 物理参数
c = 299792458  # 光速 (m/s)
lambda_radar = 0.235  # 雷达波长 (m)
baseline = 107.514  # 垂直基线 (m)
x_spacing_830 = 11.242217175 #isce运行的时候可以看的到
x_spacing_909 = 11.242217175 # 斜距方向像素间距 (m)
system_bandwidth = 23.68e6  # SAR 系统带宽 (Hz)

# 读取SAR数据
def read_gdal_array(filepath):
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset.GetRasterBand(1).ReadAsArray()

#读取配准数据
array_830 = read_gdal_array("830/saocom_20240829_105058528_QS6_D_HV_slc_rsp")
array_909 = read_gdal_array("909/saocom_20240906_105435075_QS6_D_HV_slc_rsp")
#读取视角信息
angle_830 = np.radians(read_gdal_array("830/look_angle_resize.dat"))
angle_909 = np.radians(read_gdal_array("909/look_angle_resize.dat"))
#读取坡度信息
slope_angle_830 = np.radians(read_gdal_array("830/slope_angle_resize.dat"))
slope_angle_909 = np.radians(read_gdal_array("909/slope_angle_resize.dat"))

# 计算斜距对应的 wavenumber shift
rows_830, cols_830 = array_830.shape
rows_909, cols_909 = array_909.shape
f_830 = np.zeros((rows_830, cols_830))
f_909 = np.zeros((rows_909, cols_909))
print(array_830.shape, array_909.shape)


range_near_830 = 696371.072  # 最近距离  isce运行日志里面有
range_near_909 = 696389.97726431070
#计算
def f_fix(shape, angle,slope, range_near,x):
    f_ = np.zeros(shape)
    rows, cols = shape
    for i in range(angle.shape[0]):
        for j in range(angle.shape[1]):  # j 的范围是 0 到 8083
            r_ = range_near + j * x
            denominator = lambda_radar * np.tan(angle[i, j] - slope[i, j]) * r_ * 1e6
            if denominator != 0:
                f_[i, j] = -(c * baseline) / denominator
            else:
                f_[i, j] = 0  # 或者其他合适的处理方式
    return f_

f_830 = f_fix(array_830.shape, angle_830,slope_angle_830,range_near_830,x_spacing_830)
f_909 = f_fix(array_909.shape, angle_909,slope_angle_909,range_near_909,x_spacing_909)

# 计算 FFT
fft_830 = fftshift(fft2(array_830))
fft_909 = fftshift(fft2(array_909))

#高斯滤波
def bandpass_filter(shape, f_array, bandwidth):
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.float32)
    ccol = cols // 2  # 中心列
    sigma = bandwidth / 4  # 设定滤波器带宽
    for i in range(rows):
        for j in range(cols):
            shift = f_array[i, j] * cols / bandwidth  # 计算偏移量
            mask[i, :] += np.exp(-((np.arange(cols) - (ccol - shift)) ** 2) / (2 * sigma ** 2))

    # 归一化
    mask = mask / np.max(mask)
    return mask


filter_830 = bandpass_filter(array_830.shape, f_830, system_bandwidth)
filter_909 = bandpass_filter(array_909.shape, f_909, system_bandwidth)

# 应用滤波器
filtered_fft_830 = fft_830 * filter_830
filtered_fft_909 = fft_909 * filter_909

# 逆变换回时域（不取绝对值）
filtered_image_830 = ifft2(ifftshift(filtered_fft_830))
filtered_image_909 = ifft2(ifftshift(filtered_fft_909))

# 保存滤波结果为 ENVI 格式
def save_complex_as_complex(output_path, complex_array, reference_file):
    """
    将复数数据直接存储为一个复数波段
    参数：
    - output_path: 输出文件路径 (如 "filtered_image_830_complex.dat")
    - complex_array: 复数数据
    - reference_file: 参考 ENVI 文件 (用于获取地理坐标信息)
    """
    ref_dataset = gdal.Open(reference_file, gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName('ENVI')

    rows, cols = complex_array.shape

    # 创建输出文件
    out_dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_CFloat32)  # 1 个复数波段

    # 复制地理坐标信息
    out_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_dataset.SetProjection(ref_dataset.GetProjection())

    # 写入复数数据
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(complex_array)

    # 关闭数据集
    out_band.FlushCache()
    out_dataset = None
    print(f"复数数据已保存为复数格式文件: {output_path}")

# 保存 830 和 909 结果
save_complex_as_complex("filtered_image_830_HV.dat", filtered_image_830, "830/saocom_20240829_105058528_QS6_D_HH_slc_rsp")
save_complex_as_complex("filtered_image_909_HV.dat", filtered_image_909, "909/saocom_20240906_105435075_QS6_D_HH_slc_rsp")

end_time = datetime.now()
print("运行结束，时间为", datetime.now())
print("程序共计运行时间为：",(end_time - strat_time))
# plt.imshow(filter_830, cmap='gray')
# plt.title("Filter 830")
# plt.colorbar()
# plt.show()
#
# print("结束运行")
# # 结果可视化（分别显示实部和虚部）
# plt.figure(figsize=(12, 6))
#
# plt.subplot(2, 2, 1)
# plt.title("Filtered Image 830 - Real Part")
# plt.imshow(np.real(filtered_image_830), cmap='gray')
#
# plt.subplot(2, 2, 2)
# plt.title("Filtered Image 830 - Imaginary Part")
# plt.imshow(np.imag(filtered_image_830), cmap='gray')
#
# plt.subplot(2, 2, 3)
# plt.title("Filtered Image 909 - Real Part")
# plt.imshow(np.real(filtered_image_909), cmap='gray')
#
# plt.subplot(2, 2, 4)
# plt.title("Filtered Image 909 - Imaginary Part")
# plt.imshow(np.imag(filtered_image_909), cmap='gray')
#
# plt.show()

