
from PyQt5.QtWidgets import QMessageBox

def check_shapes_and_warn(*arrays):
    """
    检查所有数组的前两个维度（row, col）是否一致。
    如果不一致，弹出警告窗口。
    返回：True（全部一致）或 False（存在不一致）
    """
    shapes = [arr.shape[:2] for arr in arrays]
    first_shape = shapes[0]
    for i, shape in enumerate(shapes[1:], start=1):
        if shape != first_shape:
            QMessageBox.warning(None, "数据尺寸不匹配",
                                f"第 1 个数据尺寸（行, 列）：{first_shape}\n"
                                f"第 {i + 1} 个数据尺寸（行, 列）：{shape}\n"
                                f"数据的空间维度（行列）不一致，请检查输入。")
            return False
    return True


def check_array_valid(*arrays):
    """
    检查多个数组是否有效（非 None、是 ndarray、非空）。
    无效时弹窗提示，返回 False。
    有效时返回 True。
    """
    for i, arr in enumerate(arrays, start=1):
        if arr is None:
            QMessageBox.warning(None, "数组为空", f"第{i}个输入数组为 None，请检查输入。")
            return False
        if arr.size == 0 or any(dim == 0 for dim in arr.shape):
            QMessageBox.warning(None, "数组为空", f"第{i}个输入数组为空或包含零维度。")
            return False
    return True

def check_tree_results(tree_species, geotransform, projection):
    """
    判断树种数据及其空间信息是否有效。
    无效时弹窗提示并返回 False，有效返回 True。
    """
    if tree_species is None:
        QMessageBox.warning(None, "数据错误", "树种数据为空，请检查输入文件。")
        return False

    if tree_species.size == 0 or any(dim == 0 for dim in tree_species.shape):
        QMessageBox.warning(None, "数据错误", "树种数据为空或尺寸不正确。")
        return False

    if geotransform is None:
        QMessageBox.warning(None, "数据错误", "地理变换信息为空。")
        return False

    if projection is None or not isinstance(projection, str) or projection.strip() == "":
        QMessageBox.warning(None, "数据错误", "投影信息为空或格式不正确。")
        return False

    return True
