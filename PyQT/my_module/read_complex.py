
import numpy as np
from osgeo import gdal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)

def read_complex_tif(path):
    """
    使用 gdal 读取复数型 TIF 图像，假设通道1为实部，通道2为虚部。
    返回：复数 numpy 数组
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"无法打开文件: {path}")
    if dataset.RasterCount < 2:
        raise ValueError(f"{path} 不包含复数所需的两个波段（实部和虚部）")

    real = dataset.GetRasterBand(1).ReadAsArray()
    imag = dataset.GetRasterBand(2).ReadAsArray()

    dataset = None  # 显式关闭
    return real + 1j * imag


def read_complex_bin(path, shape, dtype=np.complex128):
    """
    读取复数型二进制文件（.bin），假设数据以 NumPy 复数格式存储，
    实部和虚部交替排列。

    参数：
        path: str，bin文件路径
        shape: tuple，数据的形状，例如 (H, W, 3, 3)
        dtype: numpy dtype，复数数据类型，默认 np.complex64

    返回：
        复数 numpy 数组，形状为 shape
    """
    data = np.fromfile(path, dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(f"数据大小 {data.size} 与预期形状 {shape} 不匹配")
    data = data.reshape(shape)
    return data

def read_tree_tif(tif_path):
    """
    读取一个 GeoTIFF 文件，返回数据数组、地理变换、投影信息。

    参数:
        tif_path (str): TIF 文件路径

    返回:
        data (np.ndarray): 第一波段的数组数据
        geotransform (tuple): 地理变换信息
        projection (str): 投影字符串
    """
    dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"无法打开文件: {tif_path}")

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    return data, geotransform, projection


def read_int_tif(path):
    """
    读取单波段整型 TIF 图像

    参数:
        path: TIF 文件路径

    返回:
        int 类型 numpy 数组 (height, width)

    异常:
        ValueError: 如果不是单波段或不是整型数据
    """
    # 打开数据集
    dataset = gdal.Open(path)
    if dataset is None:
        raise IOError(f"无法打开文件: {path}")

    try:
        # 验证波段数量
        if dataset.RasterCount != 1:
            raise ValueError(f"{path} 应为单波段整数图像，但包含 {dataset.RasterCount} 个波段")

        # 获取第一个波段
        band = dataset.GetRasterBand(1)

        # 验证数据类型是否为整型
        data_type = band.DataType
        if data_type not in (gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16,
                             gdal.GDT_UInt32, gdal.GDT_Int32):
            raise ValueError(f"{path} 应为整型数据，实际类型为 {gdal.GetDataTypeName(data_type)}")

        # 读取数据
        array = band.ReadAsArray()

        # 确保返回的是numpy数组 以及地理交换、投影信息
        return np.array(array, dtype=array.dtype),dataset.GetGeoTransform(),dataset.GetProjection()
    finally:
        # 确保数据集被正确关闭
        dataset = None


def read_single_band_float_tif(path, parent=None):
    """
    读取单波段浮点型 TIF，返回二维 numpy array。
    如果出现错误，返回 None 或引发异常。

    参数:
        path: 文件路径
        parent: 父窗口（用于显示错误消息）

    返回:
        numpy.ndarray: 二维浮点数组
        None: 如果出现错误
    """
    try:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"无法打开文件：{path}")

        if dataset.RasterCount != 1:
            raise RuntimeError(f"{path} 应为单波段图像，但实际有 {dataset.RasterCount} 个波段。")

        band = dataset.GetRasterBand(1)
        data_type = band.DataType

        if data_type not in (gdal.GDT_Float32, gdal.GDT_Float64):
            raise RuntimeError(
                f"{path} 第 1 个波段应为浮点型，但实际为 {gdal.GetDataTypeName(data_type)}"
            )

        arr = band.ReadAsArray()

        # 确保返回的是 numpy 数组
        result = arr.astype(np.float32 if data_type == gdal.GDT_Float32 else np.float64)
        if not isinstance(result, np.ndarray):
            raise RuntimeError("读取的数据不是预期的 numpy 数组格式")

        return result

    except Exception as e:
        if parent:
            QMessageBox.critical(parent, "错误", str(e))
        return None

def read_two_band_float_tif(path, parent=None):
    """
    读取双波段浮点型 TIF，返回两个二维 numpy array (band1, band2)。
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        if parent:
            QMessageBox.critical(parent, "文件错误", f"无法打开文件：\n{path}")
        return None, None

    if dataset.RasterCount != 2:
        if parent:
            QMessageBox.critical(parent, "波段错误", f"{path} 应为双波段图像，但实际有 {dataset.RasterCount} 个波段。")
        return None, None

    def check_and_read(band_index):
        band = dataset.GetRasterBand(band_index)
        data_type = band.DataType
        if data_type not in (gdal.GDT_Float32, gdal.GDT_Float64):
            if parent:
                QMessageBox.critical(parent, "类型错误",
                                     f"{path} 第 {band_index} 个波段应为浮点型，但实际为 {gdal.GetDataTypeName(data_type)}")
            return None
        arr = band.ReadAsArray()
        return arr.astype(np.float32 if data_type == gdal.GDT_Float32 else np.float64)

    band1 = check_and_read(1)
    band2 = check_and_read(2)
    if band1 is None or band2 is None:
        return None, None

    return band1, band2
