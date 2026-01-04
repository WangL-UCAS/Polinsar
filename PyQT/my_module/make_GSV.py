import joblib
import numpy as np
import sys
import os
import pandas as pd
from scipy.ndimage import uniform_filter, uniform_filter1d

# 模型路径拼接函数
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# 加载模型
model_path = resource_path(r"my_module\tree_xgb_rf_to_dbh.joblib")
model_bundle = joblib.load(model_path)
xgb_model = model_bundle["xgb_model"]
rf_model = model_bundle["rf_model"]
weight_xgb = model_bundle["weight_xgb"]
weight_rf = model_bundle["weight_rf"]

def preprocess_inputs(height, tree_species, window_size=3):
    """
    仅进行平滑处理（无多视），并构造特征输入。
    """
    height = np.array(height, dtype=np.float32)
    tree_species = np.array(tree_species, dtype=np.float32)

    # 原始尺寸
    orig_shape = height.shape

    # 原始无效掩膜（后续恢复用）
    original_invalid_mask = (height <= 0) | (tree_species < 1) | (tree_species > 7)

    # 2D 平滑
    height_smooth = uniform_filter(height, size=window_size)
    tree_spe_smooth = uniform_filter(tree_species, size=window_size)

    # 构造掩膜
    valid_mask = (height_smooth > 0) & (tree_spe_smooth >= 1) & (tree_spe_smooth <= 7)
    volume = np.zeros_like(height_smooth, dtype=np.float32)

    if not np.any(valid_mask):
        return volume, orig_shape, original_invalid_mask

    # 1D 平滑
    h_valid = uniform_filter1d(height_smooth[valid_mask], size=window_size)
    s_valid = tree_spe_smooth[valid_mask].astype(np.uint8)

    # 构造特征
    s_cat = pd.Series(s_valid).astype("category")
    df = pd.DataFrame({
        "ETH": h_valid,
        "TreeSpecies": s_cat
    })
    df["ETH_TreeInteraction"] = df["ETH"] * df["TreeSpecies"].cat.codes

    return df, volume, valid_mask, orig_shape, original_invalid_mask


def make_GSV(height, tree_species):
    df, volume, valid_mask, orig_shape, original_invalid_mask = preprocess_inputs(height, tree_species)

    # 模型预测
    pred_xgb = xgb_model.predict(df)
    pred_rf = rf_model.predict(df)
    gsv = weight_xgb * pred_xgb + weight_rf * pred_rf

    # 填入有效像素位置
    volume[valid_mask] = gsv

    # 恢复无效位置为 0
    volume[original_invalid_mask] = np.nan

    return volume
