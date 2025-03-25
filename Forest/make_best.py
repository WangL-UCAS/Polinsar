import cmath
import numpy as np

def make_T6_and_MAX(array_pauli_mask, array_pauli_T_mask, array_pauli_slave, array_pauli_T_slave, row, col):
    """
    计算 T11, T22, Ω12，并计算最优复相干系数
    """

    # 初始化 T6 矩阵
    T11 = np.zeros((row, col, 3, 3), dtype=complex)
    T22 = np.zeros((row, col, 3, 3), dtype=complex)
    Omaga12 = np.zeros((row, col, 3, 3), dtype=complex)

    # 初始化 3 个复相干系数矩阵
    Y_MAX = np.zeros((row, col), dtype=complex)
    Y_MID = np.zeros((row, col), dtype=complex)
    Y_END = np.zeros((row, col), dtype=complex)

    for i in range(row):
        for j in range(col):
            # 计算 T11, T22, Ω12
            T11[i, j] = array_pauli_mask[i, j] @ array_pauli_T_mask[i, j]
            T22[i, j] = array_pauli_slave[i, j] @ array_pauli_T_slave[i, j]
            Omaga12[i, j] = array_pauli_mask[i, j] @ array_pauli_T_slave[i, j]

            # 计算 T11 和 T22 的逆矩阵
            T11_inv = np.linalg.inv(T11[i, j])
            T22_inv = np.linalg.inv(T22[i, j])

            # 构造特征矩阵
            Matrix1 = T22_inv @ Omaga12[i, j].conj().T @ T11_inv @ Omaga12[i, j].conj().T
            Matrix2 = T11_inv @ Omaga12[i, j] @ T22_inv @ Omaga12[i, j].conj().T

            # 计算特征值和特征向量
            eig_values1, eig_vectors1 = np.linalg.eig(Matrix1)
            eig_values2, eig_vectors2 = np.linalg.eig(Matrix2)

            # 从大到小排序特征值
            sorted_indices = np.argsort(eig_values1)[::-1]

            # 计算归一化复度角
            Angle_max = cmath.phase(eig_vectors1[:, sorted_indices[0]].conj().T @ eig_vectors2[:, sorted_indices[0]])
            Angle_mid = cmath.phase(eig_vectors1[:, sorted_indices[1]].conj().T @ eig_vectors2[:, sorted_indices[1]])
            Angle_end = cmath.phase(eig_vectors1[:, sorted_indices[2]].conj().T @ eig_vectors2[:, sorted_indices[2]])

            # 计算最优复相干系数
            Y_MAX[i, j] = np.sqrt(eig_values1[sorted_indices[0]]) * np.exp(-1j * Angle_max)
            Y_MID[i, j] = np.sqrt(eig_values1[sorted_indices[1]]) * np.exp(-1j * Angle_mid)
            Y_END[i, j] = np.sqrt(eig_values1[sorted_indices[2]]) * np.exp(-1j * Angle_end)

    print("--------- T11, T22, Ω12 计算完成，并已计算最优复相干系数 -----------")
    return T11, T22, Omaga12, Y_MAX, Y_MID, Y_END
