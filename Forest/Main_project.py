from colorsys import hsv_to_rgb

from osgeo import gdal
from datetime import datetime
import make_pauli
import make_best
import numpy as np
import input_data
import make_Kz
import math
import RangeSFilter
import make_ground
from scipy.ndimage import gaussian_filter , uniform_filter, median_filter
import Rovg_not_parrel
import Rvog
import Save_Height_envi
import ground_remove
import Geo
import h5py
import Filter
from Forest.Orbit import functions as fn, modules as m
import mlutilooking as mu
def main():
    start_time = datetime.now()

    # 配置参数
    Polarization = ["HH", "HV", "VH", "VV"]
    lambda_radar = 0.235  # 雷达波长 (m)
    baseline = 1114.848# 垂直基线 (m)
    print("程序启动时间为：", start_time)
    #
    # 轨道校正
    print("开始轨道校正...")
    master = r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\903\HH\SAO1B_20240901_HH"
    slave = r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\911\HH\SAO1A_20240909_HH"

    data_lat = gdal.Open(r'E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\data\lat.dat', gdal.GA_ReadOnly)
    lat = data_lat.GetRasterBand(1).ReadAsArray()
    data_lon = gdal.Open(r'E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\data\lon.dat', gdal.GA_ReadOnly)
    lon = data_lon.GetRasterBand(1).ReadAsArray()

    """ 这里只有FP-1需要改成tif格式"""
    height_z = gdal.Open(r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\data\z.tif", gdal.GA_ReadOnly)
    height_z = height_z.GetRasterBand(1).ReadAsArray().astype(np.float64)

    a = 6378137.0  # 长半轴 (m)
    e2 = 6.69437999014e-3
    lon, lat = np.radians(lon), np.radians(lat)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    X = (N + height_z) * np.cos(lat) * np.cos(lon)
    Y = (N + height_z) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + height_z) * np.sin(lat)
    #coh : cc
    coh = gdal.Open(r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\data\FP-1-903-911_cc", gdal.GA_ReadOnly)
    coh = coh.GetRasterBand(1).ReadAsArray()
    #fas : fint
    fas = gdal.Open(r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\data\FP-1-903-911_fint", gdal.GA_ReadOnly)
    fas = fas.GetRasterBand(1).ReadAsArray()
    fas = np.angle(fas)
    #    row_kz,col_kz  是没有经过多视处理的行列号，这里的轨道校正和计算kz都是没有多视的数据结果
    row_kz,col_kz = fas.shape
    m1, parm1 = fn.read_params_gmt(master)
    s1, pars1 = fn.read_params_gmt(slave)
    correct = m.mainCorrector(X, Y, Z, coh, fas, m1, s1, 4, 4, ramp=True).runCorrector()
    print("垂直基线改进量",correct[1])
    baseline = baseline + correct[1]
    print("计算轨道相位改进量：correct_B")
    correct_B = mu.multilook_float_numba(correct[3],2,4)
    correct_orb = np.exp(1j * correct_B)
    # #TODO correct_orb 多视
    del correct,fas,coh,X, Y, Z,m1, s1

    range_values = [696371.080457656644284725189209,703845.281176169170066714286804,711315.734488956630229949951172]
    angle = gdal.Open(r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\data\incident_angle.dat",gdal.GA_ReadOnly)
    angle = angle.GetRasterBand(1).ReadAsArray()
    angle = np.radians(angle)  # 转换弧度
    print("开始计算kz")
    kz = make_Kz.make_kz(baseline, lambda_radar, range_values, angle,row_kz,col_kz)
    print("kz计算结束")
    #TODO kz 多视
    kz = mu.multilook_float_numba(kz,2,4)
    data_folders = {
        "903": r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\903",
        "911": r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\911"
    }
    #E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\911\911-VH
    # 读取数据
    print("开始读取数据...")
    array_HH_master, array_VV_master, array_HV_master, array_VH_master = input_data.read_data_from_file(data_folders["903"], '903', Polarization)
    array_HH_slave, array_VV_slave, array_HV_slave, array_VH_slave = input_data.read_data_from_file(data_folders["911"], '911', Polarization)
    row, col = array_VH_slave.shape
    #对角度进行多视

    angle = mu.multilook_float_numba(angle,2,4)
    # Z = mu.multilook_float_numba(height_z,2,4)
    #
    # print("数据读取完成，行列数为：", row, col)
    # print("开始频谱滤波")
    # array_HH_slave, array_VV_slave, array_HV_slave, array_VH_slave = RangeSFilter.process_all_pols(
    #     array_HH_master, array_VV_master, array_HV_master, array_VH_master,
    #     array_HH_slave, array_VV_slave, array_HV_slave, array_VH_slave,
    #     dem_array=Z,  # 你的DEM浮点数组
    #     inc_array=angle,  # 你的入射角度数组，单位角度
    #     wavelength=0.234,
    #     baseline=baseline,
    #     pixel_spacing=15.5859739317642596034829693963
    # )
    # del Z,height_z
    # 计算 Pauli 分解
    print("开始计算 Pauli 分解...")
    array_master_k1, array_master_k1_T = make_pauli.make_pauli(
        array_HH_master, array_VV_master, array_HV_master,
        f"{data_folders['903']}/array_903_k1.h5",
        f"{data_folders['903']}/array_903_k1_T.h5",
        row, col,
        'array_903_k1',
        'array_903_k1_T'
    )
    array_slave_k2, array_slave_k2_T = make_pauli.make_pauli(
        array_HH_slave, array_VV_slave, array_HV_slave,
        f"{data_folders['911']}/array_911_k2.h5",
        f"{data_folders['911']}/array_911_k2_T.h5",
        row, col,
        'array_911_k2',
        'array_911_k2_T'
    )
    print("Pauli 分解完成。")

    # 删除不再使用的变量
    del array_HH_master, array_VV_master, array_HV_master, array_VH_master
    del array_HH_slave, array_VV_slave, array_HV_slave, array_VH_slave

    # 计算 T11 和 T22
    print("开始计算 T11 和 T22...")
    T11 = array_master_k1 @ array_master_k1_T
    T22 = array_slave_k2 @ array_slave_k2_T
    nan_cout = np.sum(np.isnan(T11))
    print("t11nan",nan_cout)
    nan_cout = np.sum(np.isnan(T22))
    if(nan_cout != 0):
        print("T22存在NaN将其转换成0")
        T22 = np.nan_to_num(T22, nan=0.0)
    print(T11[3000,1000])

    print("开始滤波")
    T11_filter = Filter.mean_filter_complex_matrix(T11,11)
    T22_filter = Filter.mean_filter_complex_matrix(T22,11)
    print(T11_filter[3000,1000])
    print("T11shape",T11.shape)
    print("T22shape",T22.shape)
    print("计算极化矩阵的平均值")
    T_Filter = (T11_filter + T22_filter) / 2

    # 删除不再使用的变量
    del T11, T22
    del array_master_k1_T
    del array_slave_k2

    with h5py.File("FP-1-903-911_ground-T11_filter.h5", 'w') as f:
        f.create_dataset("FP-1-903-911_ground-T11_filter", data=T11_filter, dtype=T11_filter.dtype)
    with h5py.File("FP-1-903-911_ground-T22_filter.h5", 'w') as f:
        f.create_dataset("FP-1-903-911_ground-T22_filter", data=T22_filter, dtype=T22_filter.dtype)
    with h5py.File("FP-1-903-911_ground-T_Filter.h5",'w') as f:
        f.create_dataset('FP-1-903-911_ground-T_Filter', data=T_Filter, dtype=T_Filter.dtype)
    print("T11 和 T22滤波结果计算完成并保存。")

    # 删除不再使用的变量
    del T11_filter, T22_filter

    # 计算 Omage12 并去除平地相位
    print("开始计算 Omage12 进行平地相位去除")
    Omage12 = array_master_k1 @ array_slave_k2_T

    sint_file = r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\data\FP-1-903-911_sint"
    #TODO 这里要对sint进行多视处理，或者处理数据的时候就做一下多视,在remove_ground里面做,在最优复想干的时候会出现NaN,inf问题，不是sint的问题
    #TODO 垂直基线是正数的时候不会出现上述问题， 但是当垂直基线是负数的时候，会出现nan或者inf的问题？ 已找到问题， 是在计算T22的时候出现计算问题，存在NAN
    Omage12_flat = ground_remove.remove_ground(sint_file, Omage12,baseline)

    with h5py.File("Omage12_FP-1-903-911_ground-flat.h5", 'w') as f:
        f.create_dataset("Omage12_FP-1-903-911_ground-flat", data=Omage12_flat, dtype=Omage12_flat.dtype)
    print("Omage12_flat 保存去除平地结果。")

    # 删除不再使用的变量
    del Omage12
    # Omage12_flat = Omage12_flat * correct_orb[:,:,np.newaxis,np.newaxis]
    # Omage12_flat = Omage12_flat
    print("轨道校正完成，并去除轨道相位偏差,开始计算滤波。")

    with h5py.File('FP-1-903-911_ground-Omage12_flat_orb.h5','w') as f:
        f.create_dataset('FP-1-903-911_ground-Omage12_flat_orb',data=Omage12_flat,dtype=Omage12_flat.dtype)

    print('保存轨道相位校正结果')
    Omage12_filter = Filter.mean_filter_complex_matrix(Omage12_flat,11)
    print("开始计算 T6 和复相干优化...")

    with h5py.File('FP-1-903-911_ground-Omage12_filter.h5','w') as f:
        f.create_dataset('FP-1-903-911_ground-Omage12_filter',data=Omage12_filter,dtype=Omage12_filter.dtype)

    del Omage12_flat
    Omage12_filter = np.nan_to_num(Omage12_filter, nan=0.0)
    Y_MAX, Y_END = make_best.pdopt(T_Filter, Omage12_filter)

    with h5py.File("FP-1-903-911_ground-Y_max.h5", 'w') as f:
        f.create_dataset("FP-1-903-911_ground-Y_max", data=Y_MAX, dtype=Y_MAX.dtype)
    with h5py.File("FP-1-903-911_ground-Y_end.h5", 'w') as f:
        f.create_dataset("FP-1-903-911_ground-Y_end", data=Y_END, dtype=Y_END.dtype)
    print("T6 和复相干优化完成。")

    # 删除不再使用的变量
    del Omage12_filter
    del T_Filter

    # 计算地表相位、kz
    with h5py.File('FP-1-903-911_ground-Y_max.h5', 'r') as f:
        Y_MAX = f['FP-1-903-911_ground-Y_max'][:]
    with h5py.File('FP-1-903-911_ground-Y_end.h5', 'r') as f:
        Y_END = f['FP-1-903-911_ground-Y_end'][:]
    row, col = Y_END.shape

    with h5py.File("FP-1-903-911_ground-kz.h5", 'w') as f:
        f.create_dataset("FP-1-903-911_ground-kz", data=kz, dtype=kz.dtype)
    gama_ground = np.zeros((2, row, col), dtype=Y_END.dtype)
    gama_ground[0, :, :] = Y_MAX
    gama_ground[1, :, :] = Y_END
    print("开始计算ground")

    slope = gdal.Open(r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\data\slope.tif",gdal.GA_ReadOnly)
    slope = slope.GetRasterBand(1).ReadAsArray()
    slope = np.radians(slope)  # 转换弧度
    slope = mu.multilook_float_numba(slope,2,4)

    '''kz的计算也需要坡度数据'''
    kz = kz * (np.sin(angle) / np.sin(angle - slope))

    ground = make_ground.groundsolver(gama_ground, kz)
    with h5py.File('FP-1-903-911_ground-ground.h5', 'w') as f:
        f.create_dataset('FP-1-903-911_ground-ground', data=ground, dtype=ground.dtype)



    print("开始计算高度")
    ground = uniform_filter(ground, size=5)
    # height, tdf, converged = Rovg_not_parrel.rvoginv(Y_MAX, ground, angle, kz,rngslope=0)
    height, tdf, converged = Rovg_not_parrel.pdopt_parrel(Y_MAX, ground, angle, kz,rngslope=slope)
    with h5py.File('height_FP-1-903-911_ground.h5','w') as f:
        f.create_dataset("height_FP-1-903-911_ground",data=height,dtype=height.dtype)
    with h5py.File('converged_FP-1-903-911_ground.h5','w') as f:
        f.create_dataset('converged_FP-1-903-911_ground',data=converged,dtype=converged.dtype)
    Out_name = "height_FP-1-903-911_ground_.dat"
    Save_Height_envi.save_float_as_envi(Out_name, height, r"E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking\903\903-HH")
    print("高度结果计算完成并保存。")
    print("开始加入地理坐标")
    lat = mu.multilook_float_numba(lat,2,4)
    lon = mu.multilook_float_numba(lon,2,4)
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    Out_name_geo = "height_FP-1-903-911_ground_geo.dat"
    out_file = r'E:\forest\Project\Forest\FP-1\FP-1-903-911-mlutilooking'

    with h5py.File('height_FP-1-903-911_ground.h5','r') as f:
        height = f['height_FP-1-903-911_ground'][:]
    print("lat shape:", lat.shape)
    print("lon shape:", lon.shape)
    print("data shape:", height.shape)
    #tr = (1.6896e-4, 1.4341e-4)  15.5865173963547984925526179723, 15.9712584421458796413162417593, units=Meters
    Geo.radar2ll_gdal(out_file,Out_name_geo,height,lat,lon)
    print("完成地理编码")
    # 删除不再使用的变量
    # del height, tdf, converged
    # del Y_MAX, Y_END, ground

    end_time = datetime.now()
    print("程序结束时间为：", end_time)
    print("程序总计用时为：", end_time - start_time)

if __name__ == '__main__':
    main()