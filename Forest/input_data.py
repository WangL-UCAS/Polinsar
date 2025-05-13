
import numpy as np
from osgeo import gdal
def read_data_from_file(base_file, date,polarizations):
    arrays = {}  # 存储各个极化通道的数据
    hv_vh_data = []  # 用于存储 HV 和 VH 数据

    for pol in polarizations:
        filepath = f"{base_file}/{date}-{pol}"
        print("检验数据路径是否正确：", filepath)

        dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
        if dataset is not None:
            # 读取时直接指定数据类型为complex64（如果数据本身是复数形式）
            arrays[pol] = dataset.GetRasterBand(1).ReadAsArray().astype(np.complex64)

            if pol in ["HV", "VH"]:
                hv_vh_data.append(arrays[pol])  # 已转换类型，直接添加
        else:
            print(f"警告: 无法打开文件 {filepath}")
            arrays[pol] = None  # 防止后续出错

    # 计算 HV 和 VH 的平均值
    if len(hv_vh_data) == 2:
        hv_vh_avg = sum(hv_vh_data) / 2
    else:
        hv_vh_avg = None
        print("警告: HV 或 VH 数据缺失，无法计算平均值")

    return arrays.get("HH"), arrays.get("VV"), hv_vh_avg, arrays.get("VH")