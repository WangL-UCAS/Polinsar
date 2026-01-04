

import h5py
import numpy
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter , uniform_filter, median_filter
import h5py
import numpy as np
import matplotlib.pyplot as plt
import region
import mlutilooking
from matplotlib.patches import Rectangle
import mlutilooking as mu


# with h5py.File('HZZ-1-rvog-909-917_ground-kz.h5', 'r') as f:
#     kz = f['HZZ-1-rvog-909-917_ground-kz'][:]  # 假设 shape (H, W, 3, 3)
#
# with h5py.File("HZZ-1-rvog-909-917_ground-Omage12_flat_orb.h5", "r") as f:
#     OM = f['HZZ-1-rvog-909-917_ground-Omage12_flat_orb'][:]
#
# with h5py.File("HZZ-1-rvog-909-917_ground-T_Filter.h5", "r") as f:
#     tm = f['HZZ-1-rvog-909-917_ground-T_Filter'][:]
#
# with h5py.File("HZZ-1-rvog-909-917_ground-ground.h5",'r') as f:
#     ground = f['HZZ-1-rvog-909-917_ground-ground'][:]
# inc = gdal.Open(r"E:\forest\Project\Forest\HZZ-1\HZZ-1-909-917-mlutilooking\data\incident_angle.dat", gdal.GA_ReadOnly)
# inc = inc.GetRasterBand(1)
# inc = inc.ReadAsArray()
# inc = mu.multilook_float_numba(inc,2,4)
# region.rvogregion_ext_sweep(tm,OM, kz, inc,ground = ground)
# ----------------------------------------------------------
# 1. 读取 lat/lon（LUT）
# lon = gdal.Open(r"E:\forest\Project\Forest\FP-1\FP-1-911-919-mlutilooking\data\lon.dat").ReadAsArray()
# lat = gdal.Open(r"E:\forest\Project\Forest\FP-1\FP-1-911-919-mlutilooking\data\lat.dat").ReadAsArray()
#
# rows, cols = lat.shape
#
# # ----------------------------------------------------------
# # 2. 读取 DEM + geotransform
# # ----------------------------------------------------------
# dem_ds = gdal.Open(r"E:\forest\Project\DEM\slope.tif")
# dem = dem_ds.ReadAsArray()
# gt = dem_ds.GetGeoTransform()
#
# dem_rows, dem_cols = dem.shape
#
# # ----------------------------------------------------------
# # 3. LUT：地理坐标 --> DEM 像素坐标（向量化）
# # ----------------------------------------------------------
# col = np.floor((lon - gt[0]) / gt[1]).astype(np.int32)
# row = np.floor((lat - gt[3]) / gt[5]).astype(np.int32)
#
# # ----------------------------------------------------------
# # 4. 生成输出
# # ----------------------------------------------------------
# dem_res = np.full((rows, cols), np.nan, dtype=np.float32)
#
# # ----------------------------------------------------------
# # 5. 有效区域掩膜（向量化）
# # ----------------------------------------------------------
# valid = (
#     (row >= 0) & (row < dem_rows) &
#     (col >= 0) & (col < dem_cols)
# )
#
# # ----------------------------------------------------------
# # 6. 批量取 DEM 值（最近邻）
# # ----------------------------------------------------------
# dem_res[valid] = dem[row[valid], col[valid]]
#
# # ----------------------------------------------------------
# # 7. 保存 TIFF
# # ----------------------------------------------------------
# driver = gdal.GetDriverByName("GTiff")
# out = driver.Create(r"E:\forest\Project\Forest\FP-1\FP-1-911-919-mlutilooking\data\slope.tif",
#                     cols, rows, 1, gdal.GDT_Float32)
# out.GetRasterBand(1).WriteArray(dem_res)
# out.FlushCache()
# out = None


with h5py.File('HZZ-1-rvog-909-917_ground-kz.h5', 'r') as f:
    data = f['HZZ-1-rvog-909-917_ground-kz'][:]  # 假设 shape (H, W, 3, 3)
plt.imshow(data, cmap='gray')
plt.show()


#
# print("数据维度:", data.shape)
#
# print("是否包含 NaN:", np.isnan(data).any())
# print("是否包含 inf:", np.isinf(data).any())
#
# # 检查是否有全零像元（常导致奇异矩阵）
# zero_mask = np.all(data == 0, axis=(-1, -2))
# print("全零像元数量:", np.sum(zero_mask))
#
# # 进一步检查奇异矩阵：determinant 接近 0
# if data.ndim >= 3 and data.shape[-1] == data.shape[-2]:
#     det = np.linalg.det(data)
#     print("determinant 中 NaN:", np.isnan(det).any())
#     print("determinant 中 INF:", np.isinf(det).any())
#     print("determinant 过小（奇异）数量:", np.sum(np.abs(det) < 1e-10))
# else:
#     print("最后两个维度不是方阵，无法检测奇异性")
# plt.imshow(data,cmap='hsv')
# plt.show()
# # 2. 读取 slope 并 multilook
# slope_ds = gdal.Open(r'E:\forest\Project\Forest\FP-2\FP-2-1005-1018-mlutilooking\data\slope.dat', gdal.GA_ReadOnly)
# slope = slope_ds.GetRasterBand(1).ReadAsArray()
#
# # 你自己写的 multilook 函数
# slope = mlutilooking.multilook_float_numba(slope, 2, 4)
#
# # 3. 找出 data 中有效数据的位置（假设无效值为0或NaN）
# mask = np.isfinite(data) & (data != 0)
# rows, cols = np.where(mask)
#
# # 如果没有有效数据，提前退出
# if len(rows) == 0:
#     print("No valid data found.")
#     exit()
#
# # 4. 获取最小外包矩形
# row_min, row_max = rows.min(), rows.max()
# col_min, col_max = cols.min(), cols.max()
#
# # 5. 裁剪 data 和 slope
# data_crop = data[row_min:row_max+1, col_min:col_max+1]
# slope_crop = slope[row_min:row_max+1, col_min:col_max+1]
# data_crop = np.angle(data_crop)
# data_crop = uniform_filter(data_crop, size=5)
#
# # 区域框选位置（根据你之前的图像）
# y_min, y_max = 3400, 3540
# x_min, x_max = 300, 440
#
# # 放大区域数据
# zoomed_data = data_crop[y_min:y_max, x_min:x_max]
# zoomed_slope = slope_crop[y_min:y_max, x_min:x_max]
#
# # 开始画图
# fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
#
# # ---- 原始相位图 ----
# im0 = axs[0, 0].matshow(data_crop, cmap='hsv', interpolation='none', vmin=-np.pi, vmax=np.pi)
# axs[0, 0].add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
#                               edgecolor='red', facecolor='none', li2idth=1.5))
# axs[0, 0].set_title("Topographic Phase", fontsize=13)
# axs[0, 0].tick_params(labelsize=8)
# fig.colorbar(im0, ax=axs[0, 0], shrink=0.7)
#
# # ---- 原始坡度图 ----
# im1 = axs[0, 1].matshow(slope_crop, cmap='terrain', interpolation='none')
# axs[0, 1].add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
#                               edgecolor='red', facecolor='none', li2idth=1.5))
# axs[0, 1].set_title("Slope", fontsize=13)
# axs[0, 1].tick_params(labelsize=8)
# fig.colorbar(im1, ax=axs[0, 1], shrink=0.7)
#
# # ---- 放大相位图（小图）----
# im2 = axs[1, 0].imshow(zoomed_data, cmap='hsv', interpolation='none')
# axs[1, 0].set_title("Zoomed Phase", fontsize=12)
# axs[1, 0].tick_params(labelsize=7)
# fig.colorbar(im2, ax=axs[1, 0], shrink=0.7)
#
# # ---- 放大坡度图（小图）----
# im3 = axs[1, 1].matshow(zoomed_slope, cmap='terrain', interpolation='none')
# axs[1, 1].set_title("Zoomed Slope", fontsize=12)
# axs[1, 1].tick_params(labelsize=7)
# fig.colorbar(im3, ax=axs[1, 1], shrink=0.7)
#
# plt.tight_layout()
# plt.show()

#
# fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
#
# # 左图：Topographic Phase
# im0 = axs[0].matshow(np.angle(data_crop), cmap='hsv', interpolation='none')
# axs[0].set_title("Topographic Phase", fontsize=16)
# fig.colorbar(im0, ax=axs[0], shrink=0.8)
#
# # 设置左图坐标轴字体小一些，倾斜 45°
# axs[0].tick_params(labelsize=8)
# for label in axs[0].get_xticklabels():
#     label.set_rotation(45)
# for label in axs[0].get_yticklabels():
#     label.set_rotation(45)
#
# # 右图：Slope
# im1 = axs[1].matshow(slope_crop, cmap='terrain', interpolation='none')
# axs[1].set_title("Slope", fontsize=16)
# fig.colorbar(im1, ax=axs[1], shrink=0.8)
#
# # 设置右图坐标轴字体小一些，倾斜 45°
# axs[1].tick_params(labelsize=8)
# for label in axs[1].get_xticklabels():
#     label.set_rotation(45)
# for label in axs[1].get_yticklabels():
#     label.set_rotation(45)
#
# plt.tight_layout()
# plt.show()

''' 画Rvog 反演图 '''
# with h5py.File(r'FP-2-1005-1018_noSlope-T_Filter.h5', 'r') as f:
#     data = f['FP-2-1005-1018_noSlope-T_Filter'][:]
# with h5py.File(r'FP-2-1005-1018_noSlope-Omage12_filter.h5','r') as f:
#     om = f['FP-2-1005-1018_noSlope-Omage12_filter'][:]
# with h5py.File(r'FP-2-1005-1018_noSlope-kz.h5', 'r') as f:
#     kz = f['FP-2-1005-1018_noSlope-kz'][:]
# with h5py.File(r'FP-2-1005-1018_noSlope-ground.h5', 'r') as f:
#     ground = f['FP-2-1005-1018_noSlope-ground'][:]
# angel = gdal.Open(r'E:\forest\Project\Forest\FP-2\FP-2-1005-1018-mlutilooking\data\incident_angle.dat', gdal.GA_ReadOnly)
# angel = angel.GetRasterBand(1).ReadAsArray()
# angel = mlutilooking.multilook_float_numba(angel,2,4)
#
# plt.matshow(np.angle(ground), aspect='auto',cmap=plt.get_cmap('hsv'))
# plt.title('RvOG Ground')
# plt.colorbar()
# plt.show()
# tm = data[3000,1000]
# om = om[3000,1000]
# kz = kz[3000,1000]
# inc = angel[3000,1000]
# gt = ground[3000,1000]
#
# region.rvogregion_ext_sweep(tm,om,kz,inc,gt)

# '''画复平面的圆'''
# """
# 画复平面图像
# """
# with h5py.File('FP-2-1005-1018-Y_max.h5', 'r') as f:
#     data = f['FP-2-1005-1018-Y_max'][:]
# # Assuming you've already loaded the data as shown in your code
# points = [data[4000, 704 + i] for i in range(5)]
# real = np.array([p.real for p in points])
# imag = np.array([p.imag for p in points])
#
# # Create figure
# plt.figure(figsize=(8, 8))
# plt.scatter(real, imag, color='green', s=80, label='5 adjacent points', zorder=4)
#
# # ========= Fit regression line (imag = a * real + b) =========
# X = real.reshape(-1, 1)
# y = imag
#
# model = LinearRegression()
# model.fit(X, y)
# a = model.coef_[0]
# b = model.intercept_
#
# x0 = np.mean(real)
# y0 = a * x0 + b
# dx = 1
# dy = a
#
# # Find intersections with unit circle
# A = dx**2 + dy**2
# B = 2 * (x0 * dx + y0 * dy)
# C = x0**2 + y0**2 - 1
#
# t_vals = np.roots([A, B, C]) if A != 0 else [0, 0]
#
# intersections = [
#     (x0 + t*dx, y0 + t*dy) for t in t_vals
# ]
#
# # Draw fitted line (extended)
# plt.plot(
#     [intersections[0][0], intersections[1][0]],
#     [intersections[0][1], intersections[1][1]],
#     color='purple', linestyle='-', li2idth=1.5,
#     label='Fitted Line (Extended to Unit Circle)'
# )
#
# # Draw unit circle
# theta = np.linspace(0, 2*np.pi, 200)
# plt.plot(np.cos(theta), np.sin(theta), '--', color='grey', li2idth=1, label='Unit Circle')
#
# # ========= Add minimum enclosing ellipse =========
# # Combine coordinates
# points_array = np.column_stack((real, imag))
#
# # Compute covariance matrix
# cov = np.cov(points_array.T)
#
# # Compute eigenvalues and eigenvectors
# eigenvalues, eigenvectors = np.linalg.eigh(cov)
#
# # Order eigenvalues and eigenvectors
# order = eigenvalues.argsort()[::-1]
# eigenvalues = eigenvalues[order]
# eigenvectors = eigenvectors[:, order]
#
# # Angle of rotation
# angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
#
# # Width and height of ellipse (2 standard deviations)
# width, height = 4 * np.sqrt(eigenvalues)
#
# # Create ellipse
# ellipse = Ellipse(xy=np.mean(points_array, axis=0),
#                   width=width, height=height,
#                   angle=angle,
#                   edgecolor='blue', fc='None', lw=1.5, linestyle='--')
#
# plt.gca().add_patch(ellipse)
#
# # Set axes and title
# plt.xlabel('Real Axis', fontsize=12)
# plt.ylabel('Imaginary Axis', fontsize=12)
# plt.title('5 Complex Points with Fitted Line and Unit Circle', fontsize=14)
# plt.axhline(0, color='black', li2idth=0.5)
# plt.axvline(0, color='black', li2idth=0.5)
# plt.xlim(-1.1, 1.1)
# plt.ylim(-1.1, 1.1)
# plt.gca().set_aspect('equal', adjustable='box')
#
# # Label intersection points as 01 and 02
# for i, (x, y) in enumerate(intersections):
#     plt.scatter(x, y, color='red', s=60, zorder=5)
#     plt.text(x, y + 0.05, f'{i+1:02d}', fontsize=11, ha='center', color='red')
#
# plt.legend(loc='upper right')
# plt.show()
# ==== [Step 1] 数据准备 ====
# height = np.array([7.936,8.48,9.94,10.956,11.848,12.606,13.526,13.988,14.512,
#                    15.102,15.428,15.784,16.044,16.354,16.538,16.654,16.7,
#                    16.824,16.87,16.832,16.888,16.788,16.766,16.674,16.584,
#                    16.496,16.41,16.328,16.248])
# dbh = np.array([4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,
#                 46,48,50,52,54,56,58,60])
#
# # ==== [Step 2] 创建 Pipeline，使用最优参数 ====
# best_model = Pipeline([
#     ("poly", PolynomialFeatures(degree=3, include_bias=False)),
#     ("rf", RandomForestRegressor(
#         n_estimators=60,
#         max_depth=3,
#         min_samples_split=2,
#         min_samples_leaf=2,
#         random_state=42
#     ))
# ])
#
# # ==== [Step 3] 拟合模型 ====
# best_model.fit(height.reshape(-1, 1), dbh)
#
# # ==== [Step 4] 预测 + 评估 ====
# y_pred = best_model.predict(height.reshape(-1, 1))
# rmse = mean_squared_error(dbh, y_pred, squared=False)
# r2 = r2_score(dbh, y_pred)
#
# print("✅ 使用最优参数训练完毕")
# print(f"✅ RMSE: {rmse:.4f}")
# print(f"✅ R²: {r2:.4f}")
#
# # ==== [Step 5] 可视化 ====
# plt.figure(figsize=(8, 5))
# plt.scatter(height, dbh, color="black", label="True Data")
# plt.plot(height, y_pred, color="blue", label="Model Prediction", li2idth=2)
# plt.xlabel("Height (m)")
# plt.ylabel("DBH (cm)")
# plt.title("Final Height → DBH Model")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ==== [Step 6] 保存模型 ====
# dump(best_model, '../PyQT/my_module/HuaLei_height_to_dbh.joblib')
# print("✅ 模型已保存为 ../PyQT/my_module/_height_to_dbh.joblib")

