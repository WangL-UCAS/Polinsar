import os

import numpy as np
from osgeo import gdal
from my_module import make_best as mb
from my_module import Save_Height_tif
from my_module import check
from my_module import read_complex
from my_module import make_ground
from my_module import Rovg_not_parrel
from my_module import make_GSV
from my_module import Geo
from my_module import resample_array_with_georef
import os
import logging
import traceback
def run_gsv_inversion(
                t11_path, t22_path, omega_path, slope_path, inc_angle_path, tree_species_path,
                lon_lat_path, kz_path, output_path,progressbar):  # 增加 parent 参数用于弹窗父窗口

    # 配置日志记录
    log_file = os.path.join(output_path, 'gsv_inversion.log')
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # 读取数据文件
        logging.info("开始读取数据文件...")
        logging.info("读取坡度、入射角文件")
        progressbar.setValue(1)
        #1 读取坡度、入射角文件
        slope = read_complex.read_single_band_float_tif(slope_path)
        inc_angle = read_complex.read_single_band_float_tif(inc_angle_path)
        if not check.check_array_valid(slope, inc_angle):
            logging.debug(f"坡度 或 入射角文件读取失败")
        logging.info(f"坡度入射角读取成功")

        progressbar.setValue(2)
        #2 获取行列号，保证T11 T22 Om bin文件读取成功
        row, col = inc_angle.shape
        row_mid = row//2
        col_mid = col//2

        progressbar.setValue(3)
        #3 转换为弧度制
        logging.info("将入射角和坡度角转换为弧度...")
        inc_angle = np.radians(inc_angle)
        slope = np.radians(slope)
        logging.info("坡度结果：%s, 入射角结果：%s", slope[row_mid, col_mid], inc_angle[row_mid, col_mid])

        progressbar.setValue(4)
        #4 开始读取树种信息
        logging.info("读取树种数据...")
        tree_species, tree_species_input_geotransform, tree_species_input_projection = read_complex.read_tree_tif(
            tree_species_path)
        if not check.check_tree_results(tree_species, tree_species_input_geotransform, tree_species_input_projection):
            logging.debug(f"树种数据、地理信息、投影信息读取失败")
        logging.info(f"树种投影信息: {tree_species_input_projection}")
        logging.info(f"树种地理变换: {tree_species_input_geotransform}")

        progressbar.setValue(5)
        #5 读取经纬度信息
        logging.info(f"读取经纬度数据结果...")
        lon, lat = read_complex.read_two_band_float_tif(lon_lat_path)
        if not check.check_array_valid(lon, lat):
            logging.debug(f"经纬度信息读取失败")

        progressbar.setValue(6)
        #6 获取行列数
        row_lon, row_lat = lon.shape
        logging.info(f"经纬度读取成功，尺寸：{row_lon} x {row_lat}")

        progressbar.setValue(7)
        #7 提取四角经纬度坐标
        top_left = (lon[0, 0], lat[0, 0])
        top_right = (lon[0, -1], lat[0, -1])
        bottom_left = (lon[-1, 0], lat[-1, 0])
        bottom_right = (lon[-1, -1], lat[-1, -1])

        progressbar.setValue(8)
        #8 打印四角坐标
        logging.info(f"左上角 (Top Left):     Lon={top_left[0]:.6f}, Lat={top_left[1]:.6f}")
        logging.info(f"右上角 (Top Right):    Lon={top_right[0]:.6f}, Lat={top_right[1]:.6f}")
        logging.info(f"左下角 (Bottom Left):  Lon={bottom_left[0]:.6f}, Lat={bottom_left[1]:.6f}")
        logging.info(f"右下角 (Bottom Right): Lon={bottom_right[0]:.6f}, Lat={bottom_right[1]:.6f}")

        progressbar.setValue(9)
        #9 读取 bin 文件
        logging.info("读取 T11, T22, Omega 文件...")
        t11 = read_complex.read_complex_bin(t11_path, (row, col, 3, 3))
        t22 = read_complex.read_complex_bin(t22_path, (row, col, 3, 3))
        omega = read_complex.read_complex_bin(omega_path, (row, col, 3, 3))

        logging.info(f"t11[row_mid, col_mid]: {t11[row_mid, col_mid]}")
        logging.info(f"t22[row_mid, col_mid]: {t22[row_mid, col_mid]}")

        progressbar.setValue(10)
        #10 读取垂直波束文件
        logging.info(f"开始读取垂直波束")
        kz = read_complex.read_single_band_float_tif(kz_path)
        if not check.check_array_valid(kz):
            logging.warning(f"kz读取失败")
        logging.info(f"垂直波束读取成功")

        progressbar.setValue(11)
        #11 检查形状
        if not check.check_shapes_and_warn(t11, t22, omega, slope, inc_angle,kz):
            logging.warning("输入数据的形状不匹配")
            return

        progressbar.setValue(12)
        #12 计算 T
        T = (t11 + t22) / 2
        logging.info("开始计算最优复相干...")

        progressbar.setValue(13)
        #13 计算最优复相干
        Y_MAX, Y_END = mb.pdopt(T, omega)
        logging.info("最优复相干计算结束，开始计算地形相位...")

        gama_ground = np.zeros((2, row, col), dtype=Y_END.dtype)
        gama_ground[0, :, :] = Y_MAX
        gama_ground[1, :, :] = Y_END

        progressbar.setValue(14)
        #14 计算地形相位
        ground = make_ground.groundsolver(gama_ground, kz)
        logging.info("地形相位计算结束，开始计算高度...")

        progressbar.setValue(15)
        #15 计算高度
        height, tdf, converged = Rovg_not_parrel.rvoginv(Y_MAX, ground, inc_angle, kz, rngslope=slope)
        logging.info("高度计算结束，开始地理矫正...")
        logging.info("高度结果：%s", height[row_mid, col_mid])

        progressbar.setValue(16)
        #16 地理矫正
        height_geo,projection, geotransform = Geo.radar2ll_pr(output_path, 'height_geo.dat', height, lat, lon)
        logging.info("地理矫正结束，开始计算蓄积量...")
        logging.info(f"地理矫正后，地理信息、投影信息: {geotransform, projection}")
        progressbar.setValue(17)
        #17 获取高度和树种的数据维度
        height_row, height_col = height_geo.shape
        tree_species_row, tree_species_col = tree_species.shape

        progressbar.setValue(18)
        #18 树种重采样
        if tree_species_row == height_row and tree_species_col == height_col:
            logging.info("树种数据与高度数据行列匹配，计算蓄积量...")
            gsv = make_GSV.make_GSV(height_geo, tree_species)
        else:
            logging.info("树种数据与高度数据行列不匹配，进行重采样...")
            #TODO 重采样存在问题？
            tree_species_resample, new_geotransform, input_projection = resample_array_with_georef.resample_by_target_shape(
                tree_species, geotransform, projection, height_row, height_col)
            gsv = make_GSV.make_GSV(height_geo, tree_species_resample)

        logging.info("蓄积量计算结束，开始保存结果...")

        progressbar.setValue(19)
        #19 保存蓄积量结果
        save_path = os.path.join(output_path, 'gsv.tif')
        Save_Height_tif.save_array_as_tiff(save_path, gsv, projection, geotransform)
        logging.info(f"蓄积量结果已保存至 {save_path}")

    except Exception as e:
        # 捕获并记录错误
        logging.error(f"发生错误：{str(e)}")
        logging.error("详细错误信息：")
        logging.error(traceback.format_exc())
        logging.critical("程序执行失败，请检查日志文件获取更多信息")





