import numpy as np
import cmath

def make_T6_and_MAX(array_pauli_mask, array_pauli_T_mask, array_pauli_slave, array_pauli_T_slave, row, col):
    """
    计算 T11, T22, Ω12，并计算最优复相干系数（支持四维输入）
    输入:
        array_pauli_mask:    (row, col, 3, 1)
        array_pauli_T_mask:   (row, col, 1, 3)
        array_pauli_slave:   (row, col, 3, 1)
        array_pauli_T_slave:  (row, col, 1, 3)
    输出:
        T11, T22, Omaga12:   (row, col, 3, 3)
        Y_MAX, Y_MID, Y_END:  (row, col) 复相干系数
    """
    # 计算 T11, T22, Ω12 (直接四维矩阵乘法)
    T11 = array_pauli_mask @ array_pauli_T_mask  # (row, col, 3, 3)
    T22 = array_pauli_slave @ array_pauli_T_slave  # (row, col, 3, 3)
    Omaga12 = array_pauli_mask @ array_pauli_T_slave  # (row, col, 3, 3)

    # 计算 T11 和 T22 的逆矩阵 (逐像素求逆)
    T11_inv = np.linalg.inv(T11)  # (row, col, 3, 3)
    T22_inv = np.linalg.inv(T22)  # (row, col, 3, 3)

    # 构造特征矩阵 (向量化操作)
    Omaga12_conj_T = Omaga12.conj().swapaxes(-1, -2)  # 共轭转置: (row, col, 3, 3)
    Matrix1 = T22_inv @ Omaga12_conj_T @ T11_inv @ Omaga12  # (row, col, 3, 3)
    Matrix2 = T11_inv @ Omaga12 @ T22_inv @ Omaga12_conj_T  # (row, col, 3, 3)

    # 计算特征值和特征向量 (逐像素计算)
    eig_values1, eig_vectors1 = np.linalg.eig(Matrix1)  # eig_values1: (row, col, 3)
    eig_values2, eig_vectors2 = np.linalg.eig(Matrix2)  # eig_vectors1: (row, col, 3, 3)

    # 对特征值从大到小排序,然后将索引构建数组
    sorted_indices = np.argsort(eig_values1, axis=-1)[..., ::-1]  # (row, col, 3)

    # 获取最大、中、最小特征值的索引
    idx_max = sorted_indices[..., 0]  # (row, col)
    idx_mid = sorted_indices[..., 1]  # (row, col)
    idx_end = sorted_indices[..., 2]  # (row, col)

    # 提取对应的特征向量 (高级索引)
    batch_indices = np.arange(row)[:, None], np.arange(col)[None, :]  # (row, col) 的索引网格
    vec1_max = eig_vectors1[batch_indices[0], batch_indices[1], :, idx_max]  # (row, col, 3)
    vec1_mid = eig_vectors1[batch_indices[0], batch_indices[1], :, idx_mid]  # (row, col, 3)
    vec1_end = eig_vectors1[batch_indices[0], batch_indices[1], :, idx_end]  # (row, col, 3)
    vec2_max = eig_vectors2[batch_indices[0], batch_indices[1], :, idx_max]  # (row, col, 3)
    vec2_mid = eig_vectors2[batch_indices[0], batch_indices[1], :, idx_mid]  # (row, col, 3)
    vec2_end = eig_vectors2[batch_indices[0], batch_indices[1], :, idx_end]  # (row, col, 3)

    # 计算相位角 (向量化)
    Angle_max = np.angle((vec1_max.conj() * vec2_max).sum(axis=-1))  # (row, col)
    Angle_mid = np.angle((vec1_mid.conj() * vec2_mid).sum(axis=-1))  # (row, col)
    Angle_end = np.angle((vec1_end.conj() * vec2_end).sum(axis=-1))  # (row, col)

    # 计算最优复相干系数 (向量化)
    eig_max = np.take_along_axis(eig_values1, sorted_indices[..., 0:1], axis=-1).squeeze()  # (row, col)
    eig_mid = np.take_along_axis(eig_values1, sorted_indices[..., 1:2], axis=-1).squeeze()  # (row, col)
    eig_end = np.take_along_axis(eig_values1, sorted_indices[..., 2:3], axis=-1).squeeze()  # (row, col)

    Y_MAX = np.sqrt(eig_max) * np.exp(-1j * Angle_max)  # (row, col)
    Y_MID = np.sqrt(eig_mid) * np.exp(-1j * Angle_mid)  # (row, col)
    Y_END = np.sqrt(eig_end) * np.exp(-1j * Angle_end)  # (row, col)

    print("--------- T11, T22, Ω12 计算完成，并已计算最优复相干系数 -----------")
    return T11, T22, Omaga12, Y_MAX, Y_MID, Y_END