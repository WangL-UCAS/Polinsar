import numpy as np
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
import math
def remove_ground(om,lon,lat,master_orb_file,slave_orb_file,height=0.0):

    dim = np.shape(om)
    a = 6378137.0  # 长半轴 (m)
    e2 = 6.69437999014e-3  # 第一偏心率平方
    lambda_radar = 0.235  # 雷达波长 (m)

    lon,lat = np.radians(lon),np.radians(lat)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    X = (N + height) * np.cos(lat) * np.cos(lon)
    Y = (N + height) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + height) * np.sin(lat)
    local_ = np.stack((X, Y, Z * 0), axis=-1)  # shape = (24190, 4042, 3)

    master_orb = extract_and_interpolate_positions(master_orb_file,dim[0])
    slave_orb = extract_and_interpolate_positions(slave_orb_file,dim[0])
    print('slave_orb shape:',slave_orb.shape)

    master_orb_expanded = master_orb[:, np.newaxis, :]
    slave_orb_expanded = slave_orb[:, np.newaxis, :]  # 扩展轨道为  shape: (24190, 1, 3)

    distance_master = np.linalg.norm(local_ - master_orb_expanded, axis=2)
    distance_slave = np.linalg.norm(local_ - slave_orb_expanded,axis=2)
    ans = np.exp(-1j *-(4 * np.pi / lambda_radar) * (distance_master - distance_slave))

    """ 广播乘法？ """
    om_remove = om * ans[:, :, np.newaxis, np.newaxis]
    print('om_remove shape:',om_remove.shape)
    return om_remove


def extract_and_interpolate_positions(xml_file, m, method='cubic'):
    """
    从 SARscape XML 文件中提取轨道位置坐标（x, y, z），并对其插值生成指定数量的新坐标点。

    :param xml_file: XML 文件路径
    :param m: 插值后的点数 就是行数
    :param method: 插值方法，如 'linear', 'cubic', 'quadratic'
    :return: 插值后的 (m, 3) 坐标数组
    cubic 三次样条插值（更平滑
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'ns': 'http://www.sarmap.ch/xml/SARscapeHeaderSchema'}

    pos_x_list, pos_y_list, pos_z_list = [], [], []

    for pos_node in root.findall('.//ns:VectorOfStructsValues_pos', ns):
        pos_x = float(pos_node.find('ns:pos_x', ns).text)
        pos_y = float(pos_node.find('ns:pos_y', ns).text)
        pos_z = float(pos_node.find('ns:pos_z', ns).text)

        pos_x_list.append(pos_x)
        pos_y_list.append(pos_y)
        pos_z_list.append(pos_z)

    pos_array = np.column_stack((pos_x_list, pos_y_list, pos_z_list))

    n = pos_array.shape[0]
    original_indices = np.linspace(0, 1, n)
    target_indices = np.linspace(0, 1, m)

    interp_x = interp1d(original_indices, pos_array[:, 0], kind=method)
    interp_y = interp1d(original_indices, pos_array[:, 1], kind=method)
    interp_z = interp1d(original_indices, pos_array[:, 2], kind=method)

    x_new = interp_x(target_indices)
    y_new = interp_y(target_indices)
    z_new = interp_z(target_indices)

    interpolated_array = np.column_stack((x_new, y_new, z_new))
    return interpolated_array
# xml_file = r'E:\forest\Project\830\saocom_20240829_105058528_QS6_D_HH_slc_rsp_orb.sml'


