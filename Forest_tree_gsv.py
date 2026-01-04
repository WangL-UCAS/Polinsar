import pandas as pd
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ===== 1. 读取 Excel 并训练模型 =====
file_path = r"C:\Users\14618\Desktop\training sample_H_SV(1).xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

MEAN_H = df["MEAN H"].to_numpy().reshape(-1, 1)
MEAN_SV = df["MEAN SV"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    MEAN_H, MEAN_SV, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ===== 2. 读取 GeoTIFF =====

tif_path = r"C:\Users\14618\Desktop\画图\tif数据\fp-SLOPE_filter"  # MEAN_H 影像
with rasterio.open(tif_path) as src:
    mean_h_data = src.read(1)  # 读取第一波段
    profile = src.profile       # 保存原始地理信息

# ===== 3. 用模型预测 =====
# 找 nodata 区域
nodata_value = profile.get("nodata", None)
mask = (mean_h_data == nodata_value) if nodata_value is not None else np.isnan(mean_h_data)

# 有效值位置
valid_mask = ~mask

# 取出有效值，reshape 成 (n,1)
valid_h = mean_h_data[valid_mask].reshape(-1, 1)

# 预测
pred_sv_valid = model.predict(valid_h)

# 构造完整结果数组
pred_sv = np.full(mean_h_data.shape, np.nan, dtype=np.float32)  # 先全设为 nan
pred_sv[valid_mask] = pred_sv_valid  # 填回预测值

# 将原 nodata 区域设为 nodata
if nodata_value is not None:
    pred_sv[mask] = nodata_value

# ===== 4. 保存预测结果为 GeoTIFF =====
output_path = r"C:\Users\14618\Desktop\画图\tif数据\slope\FP-Slope-GSV.tif"
profile.update(dtype=rasterio.float32, count=1, nodata=nodata_value if nodata_value is not None else -9999)

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(pred_sv.astype(rasterio.float32), 1)

print(f"预测结果已保存到 {output_path}")