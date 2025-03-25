
## ———————————— 这里是读取数据的函数————————————————
from osgeo import gdal

def read_data_from_file(base_file, file_head, file_mid, file_end, polarizations, date_, number_id):
    arrays = {}  # 存储各个极化通道的数据
    hv_vh_data = []  # 用于存储 HV 和 VH 数据

    for pol in polarizations:
        filepath = f"{base_file}/{file_head}{date_}_{number_id}{file_mid}{pol}{file_end}"
        print("检验数据路径是否正确：", filepath)

        dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
        if dataset is not None:
            arrays[pol] = dataset.GetRasterBand(1).ReadAsArray()
            if pol in ["HV", "VH"]:
                hv_vh_data.append(arrays[pol])
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
