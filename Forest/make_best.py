import math
import time

import numpy as np
import cmath

from numpy.linalg import linalg


def make_T6_and_MAX(T11,T22,Omaga12, row, col):
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
    # 计算 T11 和 T22 的逆矩阵 (逐像素求逆)
    T11_inv = np.linalg.pinv(T11)  # (row, col, 3, 3)
    T22_inv = np.linalg.pinv(T22)  # (row, col, 3, 3)

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

    # 计算相位角 (向量化) 为后续的 w2 进行改正
    Angle_max = np.angle((vec1_max.conj() * vec2_max).sum(axis=-1))  # (row, col)
    Angle_mid = np.angle((vec1_mid.conj() * vec2_mid).sum(axis=-1))  # (row, col)
    Angle_end = np.angle((vec1_end.conj() * vec2_end).sum(axis=-1))  # (row, col)

    vec2_max_ = np.exp(-1j * Angle_max) * vec2_max
    vec2_mid_ = np.exp(-1j * Angle_mid) * vec2_mid
    vec2_end_ = np.exp(-1j * Angle_end) * vec2_end

    # 计算最优复相干系数 (向量化)
    eig_max = np.take_along_axis(eig_values1, sorted_indices[..., 0:1], axis=-1).squeeze()  # (row, col)
    eig_mid = np.take_along_axis(eig_values1, sorted_indices[..., 1:2], axis=-1).squeeze()  # (row, col)
    eig_end = np.take_along_axis(eig_values1, sorted_indices[..., 2:3], axis=-1).squeeze()  # (row, col)

    Angle_max = np.angle((vec1_max.conj() * vec2_max_).sum(axis=-1))
    Angle_mid = np.angle((vec1_mid.conj() * vec2_mid_).sum(axis=-1))
    Angle_end = np.angle((vec1_end.conj() * vec2_end_).sum(axis=-1))

    Y_MAX = np.sqrt(eig_max) * np.exp(-1j * Angle_max)  # (row, col)
    Y_MID = np.sqrt(eig_mid) * np.exp(-1j * Angle_mid)  # (row, col)
    Y_END = np.sqrt(eig_end) * np.exp(-1j * Angle_end)  # (row, col)

    print("--------- T11, T22, Ω12 计算完成，并已计算最优复相干系数 -----------")
    return Y_MAX, Y_MID, Y_END


def pdopt(tm, om, numph=30, step=50, reg=0.0, returnall=False):
    dim = np.shape(tm)

    # Matrix regularization:
    if reg > 0:
        regmat = np.zeros(dim, dtype='complex64')
        regmat[:, :] = np.eye(dim[2])
        regmat = regmat * reg * np.trace(tm, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis]
        tm = tm + regmat

        regmat = np.zeros(dim, dtype='complex64')
        regmat[:, :] = np.eye(dim[2])
        regmat = regmat * reg * np.trace(om, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis]
        om = om + regmat
        del regmat

    # Arrays to store coherence separation, and the two complex coherence values.
    cohsize = (dim[0], dim[1])  # number of az, rng pixels
    cohdiff = np.zeros(cohsize, dtype='float32')
    gammamax = np.zeros(cohsize, dtype='complex64')
    gammamin = np.zeros(cohsize, dtype='complex64')

    # Arrays to store minor axis coherences.
    mincohdiff = np.ones(cohsize, dtype='float32') * 99
    gammaminormax = np.zeros(cohsize, dtype='complex64')
    gammaminormin = np.zeros(cohsize, dtype='complex64')

    # Arrays to store polarimetric weighting vectors for the optimized coherences.
    weightsize = (dim[0], dim[1], dim[3])
    wmax = np.zeros(weightsize, dtype='complex64')
    wmin = np.zeros(weightsize, dtype='complex64')

    # Main Loop
    for Ph in np.arange(0, numph):  # loop through rotation angles
        Pr = Ph * np.pi / numph  # phase shift to be applied

        print('kapok.cohopt.pdopt | Current Progress: ' + str(
            np.round(Pr / np.pi * 100, decimals=2)) + '%. (' + time.ctime() + ')     ', end='\r')

        for az in range(0, dim[0], step):
            azend = az + step
            if azend > dim[0]:
                azend = dim[0]

            for rng in range(0, dim[1], step):
                rngend = rng + step
                if rngend > dim[1]:
                    rngend = dim[1]

                omblock = om[az:azend, rng:rngend]
                tmblock = tm[az:azend, rng:rngend]
                z12 = omblock.copy()

                # Apply phase shift to omega matrix:
                z12 = z12 * np.exp(1j * Pr)
                z12 = 0.5 * (z12 + np.rollaxis(np.conj(z12), 3, start=2))

                # Check if any pixels have singular covariance matrices.
                # If so, set those matrices to the identity, to keep an
                # exception from being thrown by linalg.inv().
                det = linalg.det(tmblock)
                ind = (det == 0)
                if np.any(ind):
                    tmblock[ind] = np.eye(dim[3])

                # Solve the eigenvalue problem:
                nu, w = linalg.eig(np.einsum('...ij,...jk->...ik', linalg.inv(tmblock), z12))

                wH = np.rollaxis(np.conj(w), 3, start=2)

                Tmp = np.einsum('...ij,...jk->...ik', omblock, w)
                Tmp12 = np.einsum('...ij,...jk->...ik', wH, Tmp)

                Tmp = np.einsum('...ij,...jk->...ik', tmblock, w)
                Tmp11 = np.einsum('...ij,...jk->...ik', wH, Tmp)

                azind = np.tile(np.arange(0, w.shape[0]), (w.shape[1], 1)).T
                rngind = np.tile(np.arange(0, w.shape[1]), (w.shape[0], 1))

                lmin = np.argmin(nu, axis=2)
                gmin = Tmp12[azind, rngind, lmin, lmin] / np.abs(Tmp11[azind, rngind, lmin, lmin])

                lmax = np.argmax(nu, axis=2)
                gmax = Tmp12[azind, rngind, lmax, lmax] / np.abs(Tmp11[azind, rngind, lmax, lmax])

                ind = (np.abs(gmax - gmin) > cohdiff[az:azend, rng:rngend])

                # If we've found the coherences with the best separation
                # so far, save them.
                if np.any(ind):
                    (azupdate, rngupdate) = np.where(ind)

                    cohdiff[az + azupdate, rng + rngupdate] = np.abs(gmax - gmin)[azupdate, rngupdate]
                    gammamax[az + azupdate, rng + rngupdate] = gmax[azupdate, rngupdate]
                    gammamin[az + azupdate, rng + rngupdate] = gmin[azupdate, rngupdate]

                    if returnall:
                        wmax[az + azupdate, rng + rngupdate, :] = np.squeeze(
                            w[azupdate, rngupdate, :, lmax[azupdate, rngupdate]])
                        wmin[az + azupdate, rng + rngupdate, :] = np.squeeze(
                            w[azupdate, rngupdate, :, lmin[azupdate, rngupdate]])

                # If returnall is True, also check if this coherence pair
                # has the smallest separation found so far.
                if returnall:
                    ind = (np.abs(gmax - gmin) < mincohdiff[az:azend, rng:rngend])

                    if np.any(ind):
                        (azupdate, rngupdate) = np.where(ind)

                        mincohdiff[az + azupdate, rng + rngupdate] = np.abs(gmax - gmin)[azupdate, rngupdate]
                        gammaminormax[az + azupdate, rng + rngupdate] = gmax[azupdate, rngupdate]
                        gammaminormin[az + azupdate, rng + rngupdate] = gmin[azupdate, rngupdate]

    print('kapok.cohopt.pdopt | Optimization complete. (' + time.ctime() + ')          ')
    if returnall:
        return gammamax, gammamin, gammaminormax, gammaminormin, wmax, wmin
    else:
        return gammamax, gammamin


def pdopt_pixel(tm, om, numph=60, reg=0.0):
    cohdiff = 0
    gammaregion = np.empty((numph * 2 + 1), dtype='complex')

    # Matrix regularization:
    if reg > 0:
        tm = tm + reg * np.trace(tm) * np.eye(3)
        om = om + reg * np.trace(om) * np.eye(3)

    for Ph in range(0, numph):  # loop through rotation angles
        Pr = Ph * np.pi / numph  # phase shift to be applied

        # Apply phase shift to omega matrix:
        z12 = om.copy() * np.exp(1j * Pr)
        z12 = 0.5 * (z12 + np.transpose(np.conj(z12)))

        # Solve the eigenvalue problem:
        nu, w = linalg.eig(np.dot(linalg.inv(tm), z12))

        wH = np.transpose(np.conj(w))

        Tmp = np.dot(om, w)
        Tmp12 = np.dot(wH, Tmp)

        Tmp = np.dot(tm, w)
        Tmp11 = np.dot(wH, Tmp)

        l = np.argmin(nu)
        gmin = Tmp12[l, l] / np.abs(Tmp11[l, l])  # min eigenvalue coherence

        l = np.argmax(nu)
        gmax = Tmp12[l, l] / np.abs(Tmp11[l, l])  # max eigenvalue coherence

        gammaregion[Ph] = gmin
        gammaregion[Ph + numph] = gmax

        if (np.abs(gmax - gmin) > cohdiff):
            cohdiff = np.abs(gmax - gmin)
            gammamax = gmax
            gammamin = gmin

    gammaregion[-1] = gammaregion[
        0]  # copy the first coherence to the end of the array, for a continuous coherence region plot

    return gammamax, gammamin, gammaregion