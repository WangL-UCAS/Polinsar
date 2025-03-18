import numpy as np
import rasterio
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.ndimage import sobel


#————————————————这里是通过los.rdr.full 和 z.rdr.full计算影像数据范围内的局部入射信息，并显示————————————————

def read_raster(file_path):
    """ 读取 ISCE2 `los.rdr` 或 `hgt.rdr` 文件 """
    with rasterio.open(file_path) as src:
        data = src.read()
        transform = src.transform
    return data, transform

def compute_los_vectors(incidence_angle, azimuth_angle):
    theta = np.radians(incidence_angle)
    phi = np.radians(azimuth_angle)

    los_x = np.sin(theta) * np.cos(phi)
    los_y = np.sin(theta) * np.sin(phi)
    los_z = np.cos(theta)

    los_vectors = np.stack((los_x, los_y, los_z), axis=0)

    # **归一化 LOS 向量**
    norm = np.linalg.norm(los_vectors, axis=0)
    norm[norm == 0] = 1  # 避免除零
    los_vectors /= norm

    return los_vectors


def compute_surface_normals(hgt_data, transform):
    """ 计算地形法向量（normal vector） """
    x_res = 7.74272683395724303778706598678 # X 方向像素分辨率
    y_res = 3.99279930026319007652091386262 # Y 方向像素分辨率
    dz_dx = sobel(hgt_data, axis=1) / (8 * x_res)
    dz_dy = sobel(hgt_data, axis=0) / (8 * y_res)

    norm_x = -dz_dx
    norm_y = -dz_dy
    norm_z = np.ones_like(dz_dx)

    norm_length = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
    norm_x /= norm_length
    norm_y /= norm_length
    norm_z /= norm_length

    return np.stack((norm_x, norm_y, norm_z), axis=0)

def compute_local_incidence_angle(los_vectors, norm_vectors):
    """ 计算局部入射角 (LIA) """
    # 确保向量归一化
    los_length = np.sqrt(los_vectors[0]**2 + los_vectors[1]**2 + los_vectors[2]**2)
    los_length[los_length == 0] = 1  # 避免除以 0
    los_vectors /= los_length

    norm_length = np.sqrt(norm_vectors[0]**2 + norm_vectors[1]**2 + norm_vectors[2]**2)
    norm_length[norm_length == 0] = 1  # 避免除以 0
    norm_vectors /= norm_length

    # 计算点积
    dot_product = (
        los_vectors[0] * norm_vectors[0] +
        los_vectors[1] * norm_vectors[1] +
        los_vectors[2] * norm_vectors[2]
    )

    # 限制 dot_product 范围，避免 arccos 计算出 NaN
    dot_product = np.clip(dot_product, -1.0, 1.0)
    print("Dot product min/max:", dot_product.min(), dot_product.max())
    # 计算局部入射角
    local_incidence_angle = np.arccos(dot_product) * 180 / np.pi

    return local_incidence_angle


# 读取 los.rdr 和 hgt.rdr 数据
los_data, _ = read_raster("los.rdr.full")
hgt_data, transform = read_raster("z.rdr.full")

# 提取入射角（通道 1）和方位角（通道 2）
incidence_angle = los_data[0]  # 通道 1
azimuth_angle = los_data[1]  # 通道 2



# 计算 LOS 向量
los_vectors = compute_los_vectors(incidence_angle, azimuth_angle)

# 计算地形法向量
norm_vectors = compute_surface_normals(hgt_data[0], transform)

# 计算局部入射角
local_incidence_angle = compute_local_incidence_angle(los_vectors, norm_vectors)

##保存为envi格式文件
def save_Angle_ENVI(output_path, incidence_angle, reference_file):
    ref_dataset = gdal.Open(reference_file, gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName('ENVI')
    rows, cols = incidence_angle.shape

    out_dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)  # 这里参数顺序修正
    out_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_dataset.SetProjection(ref_dataset.GetProjection())

    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(incidence_angle)
    out_band.FlushCache()

    out_band = None
    out_dataset = None
    print("局部入射角文件已保存:", output_path)



save_Angle_ENVI("ceshi_angle.dat",np.array(local_incidence_angle),"830/20240830_HH.dat")
# 可视化
plt.figure(figsize=(8, 6))
plt.imshow(local_incidence_angle, cmap='jet', vmin=0, vmax=90)
plt.colorbar(label="Local Incidence Angle (degrees)")
plt.title("Local Incidence Angle Map")
plt.xlabel("Range")
plt.ylabel("Azimuth")
plt.show()
