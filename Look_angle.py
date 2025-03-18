import numpy as np
from osgeo import gdal



"""
处理 LOS 文件，计算视角并保存结果。

参数:
    SC_height_start (float): 起始高度。
    SC_height_end (float): 结束高度。
    Earth_height (float): 地球高度（单位：米）。
    los_file (str): LOS 文件路径。
    reference_file (str): 参考文件路径（用于获取地理变换和投影信息）。
    output_path (str): 输出文件路径。
"""
def process_los_file(SC_height_start, SC_height_end, Earth_height, los_file, reference_file, output_path):

    # 打开影像文件
    try:
        dataset = gdal.Open(los_file, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"Could not open file: {los_file}")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    # 读取波段 1（入射角）
    band1 = dataset.GetRasterBand(1)
    incidence_angle = band1.ReadAsArray()
    rows, cols = incidence_angle.shape

    # 计算 x_pred
    x_pred = np.linspace(SC_height_start, SC_height_end, rows)

    # 计算视角
    look_angle = np.degrees(
        np.arcsin((Earth_height * np.sin(np.radians(incidence_angle))) / (Earth_height + x_pred[:, np.newaxis]))
    )

    # 保存视角到 ENVI 文件
    def save_Angle_ENVI(output_path, angle_data, reference_file):
        ref_dataset = gdal.Open(reference_file, gdal.GA_ReadOnly)
        driver = gdal.GetDriverByName('ENVI')
        rows, cols = angle_data.shape

        out_dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
        out_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
        out_dataset.SetProjection(ref_dataset.GetProjection())

        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(angle_data)
        out_band.FlushCache()

        out_band = None
        out_dataset = None
        print("局部入射角文件已保存:", output_path)

    save_Angle_ENVI(output_path, look_angle, reference_file)

# 调用函数
SC_height_start_909 = 626222.952244
SC_height_end_909 = 626357.607960
Earth_height_909 = 6371760.454002  # 地球高度（单位：米）
los_file_909 = "909/los.rdr.full"
reference_file_909 = "909/saocom_20240906_105435075_QS6_D_HH_slc_rsp"
output_path_909 = "909/look_angle.dat"

process_los_file(SC_height_start_909, SC_height_end_909, Earth_height_909, los_file_909, reference_file_909, output_path_909)

SC_height_start_830 = 626285.904776
SC_height_end_830 = 626434.292011
Earth_height_830 = 6371794.679579
los_file_830 = "830/los.rdr.full"
reference_file_830 = "830/saocom_20240829_105058528_QS6_D_VV_slc_rsp"
output_path_830 = "830/look_angle.dat"
process_los_file(SC_height_start_830,SC_height_end_830,Earth_height_830, los_file_830, reference_file_830, output_path_830)