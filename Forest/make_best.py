
import numpy as np
import numpy.linalg as linalg
import time
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.pool import Pool
import os

"""
    这个是判断nan或者inf位置的优化函数
"""
# def pdopt(tm, om, reg=0.001, step=50, numph=30, returnall=False):
#     import numpy as np
#     from numpy import linalg
#     import time
#
#     dim = np.shape(tm)
#
#     if reg > 0:
#         regmat = np.zeros(dim, dtype='complex64')
#         regmat[:, :] = np.eye(dim[2])
#         regmat = regmat * reg * np.trace(tm, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis]
#         tm = tm + regmat
#
#         regmat = np.zeros(dim, dtype='complex64')
#         regmat[:, :] = np.eye(dim[2])
#         regmat = regmat * reg * np.trace(om, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis]
#         om = om + regmat
#         del regmat
#
#     cohsize = (dim[0], dim[1])
#     cohdiff = np.zeros(cohsize, dtype='float32')
#     gammamax = np.zeros(cohsize, dtype='complex64')
#     gammamin = np.zeros(cohsize, dtype='complex64')
#
#     mincohdiff = np.ones(cohsize, dtype='float32') * 99
#     gammaminormax = np.zeros(cohsize, dtype='complex64')
#     gammaminormin = np.zeros(cohsize, dtype='complex64')
#
#     weightsize = (dim[0], dim[1], dim[3])
#     wmax = np.zeros(weightsize, dtype='complex64')
#     wmin = np.zeros(weightsize, dtype='complex64')
#
#     invalid_count = 0
#     invalid_positions = []
#
#     for Ph in np.arange(0, numph):
#         Pr = Ph * np.pi / numph
#         print('kapok.cohopt.pdopt | Current Progress: ' + str(
#             np.round(Pr / np.pi * 100, decimals=2)) + '%. (' + time.ctime() + ')     ', end='\r')
#
#         for az in range(0, dim[0], step):
#             azend = min(az + step, dim[0])
#
#             for rng in range(0, dim[1], step):
#                 rngend = min(rng + step, dim[1])
#
#                 omblock = om[az:azend, rng:rngend]
#                 tmblock = tm[az:azend, rng:rngend]
#                 z12 = omblock.copy()
#
#                 for name, mat in [('omblock', omblock), ('tmblock', tmblock)]:
#                     if np.isnan(mat).any() or np.isinf(mat).any():
#                         nan_locs = np.argwhere(np.isnan(mat))
#                         inf_locs = np.argwhere(np.isinf(mat))
#                         print(f"\nDetected invalid values in {name} at block ({az}:{azend}, {rng}:{rngend})")
#                         for loc in nan_locs:
#                             print(f"  {name} NaN at local index {tuple(loc)} => global approx. ({az + loc[0]}, {rng + loc[1]})")
#                         for loc in inf_locs:
#                             print(f"  {name} Inf at local index {tuple(loc)} => global approx. ({az + loc[0]}, {rng + loc[1]})")
#                         invalid_count += 1
#                         invalid_positions.append((az, rng))
#
#                 z12 = z12 * np.exp(1j * Pr)
#                 z12 = 0.5 * (z12 + np.rollaxis(np.conj(z12), 3, start=2))
#
#                 det = linalg.det(tmblock)
#                 ind = (det == 0)
#                 if np.any(ind):
#                     tmblock[ind] = np.eye(dim[3])
#
#                 try:
#                     tmblock_inv = linalg.inv(tmblock)
#                     has_nan = np.isnan(tmblock_inv).any()
#                     has_inf = np.isinf(tmblock_inv).any()
#                     if has_nan or has_inf:
#                         nan_locs = np.argwhere(np.isnan(tmblock_inv))
#                         inf_locs = np.argwhere(np.isinf(tmblock_inv))
#                         print(f"\nInvalid inverse tmblock at ({az}:{azend}, {rng}:{rngend})")
#                         for loc in nan_locs:
#                             print(f"  Inverse NaN at local index {tuple(loc)} => global approx. ({az + loc[0]}, {rng + loc[1]})")
#                         for loc in inf_locs:
#                             print(f"  Inverse Inf at local index {tuple(loc)} => global approx. ({az + loc[0]}, {rng + loc[1]})")
#                         invalid_count += 1
#                         invalid_positions.append((az, rng))
#                 except np.linalg.LinAlgError as e:
#                     print(f"\nLinAlgError: Cannot invert tmblock at ({az}:{azend}, {rng}:{rngend}) — {e}")
#                     invalid_count += 1
#                     invalid_positions.append((az, rng))
#                     continue
#
#                 nu, w = linalg.eig(np.einsum('...ij,...jk->...ik', tmblock_inv, z12))
#                 wH = np.rollaxis(np.conj(w), 3, start=2)
#
#                 Tmp = np.einsum('...ij,...jk->...ik', omblock, w)
#                 Tmp12 = np.einsum('...ij,...jk->...ik', wH, Tmp)
#
#                 Tmp = np.einsum('...ij,...jk->...ik', tmblock, w)
#                 Tmp11 = np.einsum('...ij,...jk->...ik', wH, Tmp)
#
#                 azind = np.tile(np.arange(0, w.shape[0]), (w.shape[1], 1)).T
#                 rngind = np.tile(np.arange(0, w.shape[1]), (w.shape[0], 1))
#
#                 lmin = np.argmin(nu, axis=2)
#                 gmin = Tmp12[azind, rngind, lmin, lmin] / np.abs(Tmp11[azind, rngind, lmin, lmin])
#
#                 lmax = np.argmax(nu, axis=2)
#                 gmax = Tmp12[azind, rngind, lmax, lmax] / np.abs(Tmp11[azind, rngind, lmax, lmax])
#
#                 ind = (np.abs(gmax - gmin) > cohdiff[az:azend, rng:rngend])
#
#                 if np.any(ind):
#                     azupdate, rngupdate = np.where(ind)
#                     cohdiff[az + azupdate, rng + rngupdate] = np.abs(gmax - gmin)[azupdate, rngupdate]
#                     gammamax[az + azupdate, rng + rngupdate] = gmax[azupdate, rngupdate]
#                     gammamin[az + azupdate, rng + rngupdate] = gmin[azupdate, rngupdate]
#
#                     if returnall:
#                         wmax[az + azupdate, rng + rngupdate, :] = np.squeeze(
#                             w[azupdate, rngupdate, :, lmax[azupdate, rngupdate]])
#                         wmin[az + azupdate, rng + rngupdate, :] = np.squeeze(
#                             w[azupdate, rngupdate, :, lmin[azupdate, rngupdate]])
#
#                 if returnall:
#                     ind = (np.abs(gmax - gmin) < mincohdiff[az:azend, rng:rngend])
#                     if np.any(ind):
#                         azupdate, rngupdate = np.where(ind)
#                         mincohdiff[az + azupdate, rng + rngupdate] = np.abs(gmax - gmin)[azupdate, rngupdate]
#                         gammaminormax[az + azupdate, rng + rngupdate] = gmax[azupdate, rngupdate]
#                         gammaminormin[az + azupdate, rng + rngupdate] = gmin[azupdate, rngupdate]
#
#     print('\nkapok.cohopt.pdopt | Optimization complete. (' + time.ctime() + ')')
#
#     print(f"\nTotal invalid blocks with NaN or Inf: {invalid_count}")
#     if invalid_count > 0:
#         print("Invalid block top-left indices:")
#         for pos in invalid_positions:
#             print(f"  az: {pos[0]}, rng: {pos[1]}")
#
#     if returnall:
#         return gammamax, gammamin, gammaminormax, gammaminormin, wmax, wmin
#     else:
#         return gammamax, gammamin



def pdopt(tm, om, numph=30, step=50, reg=0.000001, returnall=False):
    """Phase diversity coherence optimization.

    Solves an eigenvalue problem in order to find the complex coherences with
    maximum separation (|a - b|) in the complex plane.  Of these two
    coherences, one should in theory represent the coherence with the
    smallest ground contribution present in the data (the 'high' coherence).
    The other then represents the coherence with the largest ground
    contribution present in the data (the 'low' coherence).

    Arguments:
        tm (array): The polarimetric covariance (T) matrix of the data,
            with dimensions: [az, rng, num_pol, num_pol].  Note that in the
            HDF5 file, covariance matrix elements below the diagonal are
            zero-valued, in order to save disk space.  The (j,i) elements
            should therefore calculated from the complex conjugate of the
            (i,j) elements using the kapok.lib.makehermitian() function before
            the matrix is passed to this function.  Note: This should be the
            average matrix of the two tracks forming the baseline, assuming
            polarimetric stationarity.
        om (array): The polarimetric interferometric (Omega) matrix of the
            data, with dimensions [az, rng, num_pol, num_pol].
        numph (int): The number of phase shifts to calculate coherences for.
            The higher the number, the smaller the spacing of the coherences
            around the coherence region perimeter.  The smaller the number,
            the faster the computation time.  Default: 30.
        step (int): Block size (in pixels) used for linalg.eig.  Higher values
            will use more memory but can run a little faster.
            Default: 50.
        reg (float): Regularization factor.  The tm matrix is added to
            the matrix reg*Tr(tm)*I, where Tr(tm) is the trace of tm, and I
            is the identity matrix.  Similarly, the omega matrix is added to
            the matrix reg*Tr(om)*I.  This regularization reduces the spread
            of the coherence region for pixels where the backscatter is
            highly polarization dependent.
        returnall (bool): True/False flag.  Set to true to return the
            weight vectors for the optimized coherences, as well as the
            pair of minor axis coherences (optimized coherence pair with
            minimum separation in the complex plane).  Default: False.

    Returns:
        gammamax (array): The optimized coherence with the max eigenvalue.
        gammamin (array): The optimized coherence with the min eigenvalue.
        gammaminormax (array): Of the coherences with the minimum separation
            in the complex plane (e.g., along the minor axis of a elliptical
            coherence region), this will be the one with the max eigenvalue.
            Only returned if returnall == True.
        gammaminormin (array): Of the coherences with the maximum separation
            in the complex plane (e.g., along the minor axis of a elliptical
            coherence region), this will be the one with the min eigenvalue.
            Only returned if returnall == True.
        wmax (array): The weight vector for the max eigenvalue coherence, if
            returnall == True.
        wmin (array): The weight vector for the min eigenvalue coherence, if
            returnall == True.

    """
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
# def pdopt_parallel(tm, om, numph=30, step=50, reg=0.1, returnall=False, num_processes=12, batch_size=6):
#     if num_processes < 2:
#         return pdopt(tm, om, numph, step, None, None, reg, returnall)
#
#     # 创建共享内存块存放原始数据（避免重复拷贝）
#     shm_tm = SharedMemory(create=True, size=tm.nbytes)
#     shm_om = SharedMemory(create=True, size=om.nbytes)
#     np_shm_tm = np.ndarray(tm.shape, dtype=tm.dtype, buffer=shm_tm.buf)
#     np_shm_om = np.ndarray(om.shape, dtype=om.dtype, buffer=shm_om.buf)
#     np_shm_tm[:] = tm[:]
#     np_shm_om[:] = om[:]
#
#     # 预分割索引避免子进程重复计算分割点
#     split_indices = np.array_split(np.arange(tm.shape[0]), num_processes, axis=0)
#     tasks = [
#         (shm_tm.name, tm.shape, tm.dtype,
#          shm_om.name, om.shape, om.dtype,
#          indices, numph, step, reg, returnall)
#         for indices in split_indices
#     ]
#
#     # 使用进程池控制并发
#     with Pool(processes=batch_size) as pool:  # 限制同时活跃进程数
#         results = pool.starmap(_pdopt_worker, tasks)
#
#     # 清理共享内存
#     shm_tm.close()
#     shm_om.close()
#     shm_tm.unlink()
#     shm_om.unlink()
#
#     # 重组结果
#     return _reassemble_results(results, returnall)
#
#
# def _pdopt_worker(shm_tm_name, tm_shape, tm_dtype,
#                   shm_om_name, om_shape, om_dtype,
#                   indices, numph, step, reg, returnall):
#     try:
#         # 访问共享内存
#         shm_tm = SharedMemory(name=shm_tm_name)
#         shm_om = SharedMemory(name=shm_om_name)
#
#         # 直接通过索引切片获取数据（零拷贝）
#         tm_slice = np.ndarray(tm_shape, dtype=tm_dtype, buffer=shm_tm.buf)[indices]
#         om_slice = np.ndarray(om_shape, dtype=om_dtype, buffer=shm_om.buf)[indices]
#
#         # 执行计算
#         result = pdopt(tm_slice, om_slice, numph, step, None, None, reg, returnall)
#
#         # 立即关闭共享内存引用
#         shm_tm.close()
#         shm_om.close()
#
#         return result
#     except Exception as e:
#         return e
#
#
# def _reassemble_results(results, returnall):
#     # 按顺序合并结果
#     if returnall:
#         gammamax, gammamin, gammaminormax, gammaminormin, wmax, wmin = ([] for _ in range(6))
#         for res in results:
#             if isinstance(res, Exception):
#                 raise res
#             gm, gn, gmnr, gmnin, wm, wn = res
#             gammamax.append(gm)
#             gammamin.append(gn)
#             gammaminormax.append(gmnr)
#             gammaminormin.append(gmnin)
#             wmax.append(wm)
#             wmin.append(wn)
#         return (np.vstack(gammamax), np.vstack(gammamin),
#                 np.vstack(gammaminormax), np.vstack(gammaminormin),
#                 np.vstack(wmax), np.vstack(wmin))
#     else:
#         gammamax, gammamin = [], []
#         for res in results:
#             if isinstance(res, Exception):
#                 raise res
#             gm, gn = res
#             gammamax.append(gm)
#             gammamin.append(gn)
#         return np.vstack(gammamax), np.vstack(gammamin)

# def pdopt(tm, om, numph, step, result_dict, index, reg, returnall=False):
#     dim = np.shape(tm)
#
#     # Matrix regularization
#     if reg > 0:
#         regmat = np.zeros(dim, dtype='complex64')
#         regmat[:, :] = np.eye(dim[2])
#         regmat = regmat * reg * np.trace(tm, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis]
#         tm = tm + regmat
#
#
#         regmat = np.zeros(dim, dtype='complex64')
#         regmat[:, :] = np.eye(dim[2])
#         regmat = regmat * reg * np.trace(om, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis]
#         om = om + regmat
#         del regmat
#
#     cohsize = (dim[0], dim[1])
#     cohdiff = np.zeros(cohsize, dtype='float32')
#     gammamax = np.zeros(cohsize, dtype='complex64')
#     gammamin = np.zeros(cohsize, dtype='complex64')
#
#     mincohdiff = np.ones(cohsize, dtype='float32') * 99
#     gammaminormax = np.zeros(cohsize, dtype='complex64')
#     gammaminormin = np.zeros(cohsize, dtype='complex64')
#
#     weightsize = (dim[0], dim[1], dim[3])
#     wmax = np.zeros(weightsize, dtype='complex64')
#     wmin = np.zeros(weightsize, dtype='complex64')
#
#     for Ph in range(numph):
#         Pr = Ph * np.pi / numph
#         print(f'kapok.cohopt.pdopt | Current Progress: {Pr / np.pi * 100:.2f}%. ({time.ctime()})     ', end='\r')
#
#         for az in range(0, dim[0], step):
#             azend = min(az + step, dim[0])
#             for rng in range(0, dim[1], step):
#                 rngend = min(rng + step, dim[1])
#
#                 omblock = om[az:azend, rng:rngend]
#                 tmblock = tm[az:azend, rng:rngend]
#                 z12 = omblock * np.exp(1j * Pr)
#                 z12 = 0.5 * (z12 + np.rollaxis(np.conj(z12), 3, 2))
#
#                 det = linalg.det(tmblock)
#                 ind = (det == 0)
#                 if np.any(ind):
#                     tmblock[ind] = np.eye(dim[3])
#
#                 nu, w = linalg.eig(np.einsum('...ij,...jk->...ik', linalg.inv(tmblock), z12))
#                 wH = np.rollaxis(np.conj(w), 3, 2)
#
#                 Tmp12 = np.einsum('...ij,...jk->...ik', wH, np.einsum('...ij,...jk->...ik', omblock, w))
#                 Tmp11 = np.einsum('...ij,...jk->...ik', wH, np.einsum('...ij,...jk->...ik', tmblock, w))
#
#                 azind = np.tile(np.arange(w.shape[0]), (w.shape[1], 1)).T
#                 rngind = np.tile(np.arange(w.shape[1]), (w.shape[0], 1))
#
#                 lmin = np.argmin(nu, axis=2)
#                 denominator_min = np.abs(Tmp11[azind, rngind, lmin, lmin])
#                 gmin = np.divide(Tmp12[azind, rngind, lmin, lmin], denominator_min, where=denominator_min != 0,
#                                  out=np.zeros_like(denominator_min, dtype='complex64'))
#
#                 lmax = np.argmax(nu, axis=2)
#                 denominator_max = np.abs(Tmp11[azind, rngind, lmax, lmax])
#                 gmax = np.divide(Tmp12[azind, rngind, lmax, lmax], denominator_max, where=denominator_max != 0,
#                                  out=np.zeros_like(denominator_max, dtype='complex64'))
#
#                 ind = (np.abs(gmax - gmin) > cohdiff[az:azend, rng:rngend])
#                 if np.any(ind):
#                     azupdate, rngupdate = np.where(ind)
#                     cohdiff[az + azupdate, rng + rngupdate] = np.abs(gmax - gmin)[azupdate, rngupdate]
#                     gammamax[az + azupdate, rng + rngupdate] = gmax[azupdate, rngupdate]
#                     gammamin[az + azupdate, rng + rngupdate] = gmin[azupdate, rngupdate]
#
#                     if returnall:
#                         wmax[az + azupdate, rng + rngupdate] = w[azupdate, rngupdate, :, lmax[azupdate, rngupdate]]
#                         wmin[az + azupdate, rng + rngupdate] = w[azupdate, rngupdate, :, lmin[azupdate, rngupdate]]
#
#                 if returnall:
#                     ind = (np.abs(gmax - gmin) < mincohdiff[az:azend, rng:rngend])
#                     if np.any(ind):
#                         azupdate, rngupdate = np.where(ind)
#                         mincohdiff[az + azupdate, rng + rngupdate] = np.abs(gmax - gmin)[azupdate, rngupdate]
#                         gammaminormax[az + azupdate, rng + rngupdate] = gmax[azupdate, rngupdate]
#                         gammaminormin[az + azupdate, rng + rngupdate] = gmin[azupdate, rngupdate]
#
#     print(f'kapok.cohopt.pdopt | Optimization complete. ({time.ctime()})          ')
#
#     if result_dict is not None:
#         if returnall:
#             result_dict[index] = (gammamax, gammamin, gammaminormax, gammaminormin, wmax, wmin)
#         else:
#             result_dict[index] = (gammamax, gammamin)
#         return result_dict
#     elif returnall:
#         return gammamax, gammamin, gammaminormax, gammaminormin, wmax, wmin
#     else:
#         return gammamax, gammamin


def pdopt_parallel(tm, om, numph=30, step=50, reg=0.00001, returnall=False, num_processes=10):
    if num_processes < 2:
        return pdopt(tm, om, numph, step, None, None, reg, returnall)

    # 分割输入数据
    splits_tm = np.array_split(tm, num_processes, axis=0)
    splits_om = np.array_split(om, num_processes, axis=0)

    manager = mp.Manager()
    result_dict = manager.dict()
    #一次运行十个进程：
    processes = []  # 存储所有进程

    # 一次性启动所有进程（最多10个）
    # for j in range(num_processes):
    #     p = mp.Process(
    #         target=pdopt,
    #         args=(splits_tm[j], splits_om[j], numph, step, result_dict, j, reg, returnall)
    #     )
    #     processes.append(p)
    #     p.start()
    #
    # # 等待所有进程完成
    # for p in processes:
    #     p.join()

    # 每次运行5个进程
    for i in range(0, num_processes, 5):
        processes = []
        for j in range(i, min(i + 5, num_processes)):
            p = mp.Process(
                target=pdopt,
                args=(splits_tm[j], splits_om[j], numph, step, result_dict, j, reg, returnall)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    # 收集结果
    if returnall:
        gammamax, gammamin, gammaminormax, gammaminormin, wmax, wmin = ([] for _ in range(6))
        for i in range(num_processes):
            gm, gn, gmnr, gmnin, wm, wn = result_dict[i]
            gammamax.append(gm)
            gammamin.append(gn)
            gammaminormax.append(gmnr)
            gammaminormin.append(gmnin)
            wmax.append(wm)
            wmin.append(wn)
        return (np.vstack(gammamax), np.vstack(gammamin),
                np.vstack(gammaminormax), np.vstack(gammaminormin),
                np.vstack(wmax), np.vstack(wmin))
    else:
        gammamax, gammamin = [], []
        for i in range(num_processes):
            gm, gn = result_dict[i]
            gammamax.append(gm)
            gammamin.append(gn)
        return np.vstack(gammamax), np.vstack(gammamin)



# def make_T6_and_MAX(T11, T22, Omaga12, row, col, block_size=500):
#     """
#     优化后的函数，通过分块处理解决内存问题，并确保位置准确性
#     """
#     # ================== 1. 降低精度 ==================
#     T11 = T11.astype(np.complex64)
#     T22 = T22.astype(np.complex64)
#     Omaga12 = Omaga12.astype(np.complex64)
#
#     # ================== 2. 分块计算伪逆 ==================
#     def block_pinv(arr):
#         inv_arr = np.empty_like(arr)
#         for i in range(0, arr.shape[0], block_size):
#             for j in range(0, arr.shape[1], block_size):
#                 # 处理边缘块
#                 i_end = min(i + block_size, arr.shape[0])
#                 j_end = min(j + block_size, arr.shape[1])
#                 block = arr[i:i_end, j:j_end]
#                 inv_block = np.linalg.pinv(block)
#                 inv_arr[i:i_end, j:j_end] = inv_block  # 确保位置正确
#         return inv_arr
#
#     T11_inv = block_pinv(T11)
#     T22_inv = block_pinv(T22)
#     print("------- 分块计算伪逆完成 -----")
#
#     # ================== 3. 分块构造特征矩阵 ==================
#     Omaga12_conj_T = Omaga12.conj().swapaxes(-1, -2)
#
#     def block_matmul(A, B, C, D):
#         result = np.empty(A.shape[:-2] + (3, 3), dtype=np.complex64)
#         for i in range(0, A.shape[0], block_size):
#             for j in range(0, A.shape[1], block_size):
#                 i_end = min(i + block_size, A.shape[0])
#                 j_end = min(j + block_size, A.shape[1])
#                 a = A[i:i_end, j:j_end]
#                 b = B[i:i_end, j:j_end]
#                 c = C[i:i_end, j:j_end]
#                 d = D[i:i_end, j:j_end]
#                 result[i:i_end, j:j_end] = a @ b @ c @ d
#         return result
#
#     Matrix1 = block_matmul(T22_inv, Omaga12_conj_T, T11_inv, Omaga12)
#     Matrix2 = block_matmul(T11_inv, Omaga12, T22_inv, Omaga12_conj_T)
#     print("------- 分块构造特征矩阵完成 -----")
#
#     # ================== 4. 分块计算特征值 ==================
#     def block_eig(matrix):
#         eig_values = np.empty(matrix.shape[:-2] + (3,), dtype=np.complex64)
#         eig_vectors = np.empty_like(matrix)
#         for i in range(0, matrix.shape[0], block_size):
#             for j in range(0, matrix.shape[1], block_size):
#                 i_end = min(i + block_size, matrix.shape[0])
#                 j_end = min(j + block_size, matrix.shape[1])
#                 block = matrix[i:i_end, j:j_end]
#                 vals, vecs = np.linalg.eig(block)
#                 eig_values[i:i_end, j:j_end] = vals
#                 eig_vectors[i:i_end, j:j_end] = vecs
#         return eig_values, eig_vectors
#
#     eig_values1, eig_vectors1 = block_eig(Matrix1)
#     eig_values2, eig_vectors2 = block_eig(Matrix2)
#     print("------- 分块计算特征值完成 -----")
#
#     # ================== 5. 特征值排序与索引处理 ==================
#     sorted_indices = np.argsort(eig_values1, axis=-1)[..., ::-1]
#     idx_max = sorted_indices[..., 0]
#     idx_end = sorted_indices[..., 2]
#
#     # ================== 6. 分块提取特征向量 ==================
#     def block_extract(eig_vectors, indices):
#         result = np.empty(eig_vectors.shape[:-2] + (3,), dtype=np.complex64)
#         for i in range(0, eig_vectors.shape[0], block_size):
#             for j in range(0, eig_vectors.shape[1], block_size):
#                 i_end = min(i + block_size, eig_vectors.shape[0])
#                 j_end = min(j + block_size, eig_vectors.shape[1])
#                 block = eig_vectors[i:i_end, j:j_end]
#                 idx_block = indices[i:i_end, j:j_end]
#                 result[i:i_end, j:j_end] = np.take_along_axis(
#                     block, idx_block[..., None, None], axis=-1
#                 ).squeeze()
#         return result
#
#     vec1_max = block_extract(eig_vectors1, idx_max)
#     vec1_end = block_extract(eig_vectors1, idx_end)
#     vec2_max = block_extract(eig_vectors2, idx_max)
#     vec2_end = block_extract(eig_vectors2, idx_end)
#     print("------- 分块提取特征向量完成 -----")
#
#     # ================== 7. 相位角计算 ==================
#     Angle_max = np.angle((vec1_max.conj() * vec2_max).sum(axis=-1))
#     Angle_end = np.angle((vec1_end.conj() * vec2_end).sum(axis=-1))
#     print("------- 相位角计算完成 -----")
#
#     # ================== 8. 最终结果计算 ==================
#     Y_MAX = np.sqrt(np.take_along_axis(eig_values1, idx_max[..., None], -1).squeeze()) * np.exp(-1j * Angle_max)
#     Y_END = np.sqrt(np.take_along_axis(eig_values1, idx_end[..., None], -1).squeeze()) * np.exp(-1j * Angle_end)
#
#     print("--------- 全部分块计算完成 -----------")
#     return Y_MAX, Y_END
