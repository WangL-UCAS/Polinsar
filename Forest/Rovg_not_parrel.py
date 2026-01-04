import collections
import time

import numpy as np
from numpy import linalg
from concurrent.futures import ThreadPoolExecutor
# def rvoginv(gamma, phi, inc, kz, ext=None, tdf=0.7, mu=0.35, rngslope=0.0,
#             mask=None, limit2pi=True, hv_min=2, hv_max=45.0, hv_step=0.01,
#             ext_min=0.01151, ext_max=0.02875, silent=False):

def pdopt_parrel(gamma, phi, inc, kz,
                 ext=None, tdf=0.7, mu=0.35, rngslope=0.0,
                 mask=None, limit2pi=True,
                 hv_min=2, hv_max=45.0, hv_step=0.01,
                 ext_min=0.01151, ext_max=0.02875,
                 silent=False, parrel_idex=5):
    """
    并行调用 rvoginv()，把大矩阵按行分成 parrel_idex 份并行处理。
    """

    # ------ 输入矩阵尺寸 ------
    rows, cols = gamma.shape

    # 若 mask 是 None，构造一个 True mask
    if mask is None:
        mask = np.ones_like(gamma, dtype=bool)

    # ====== 函数：处理一个子矩阵 ======
    def worker(start_row, end_row):
        """
        执行 rvoginv，处理行块 [start_row, end_row)
        """
        # 切片
        gamma_sub = gamma[start_row:end_row, :]
        phi_sub = phi[start_row:end_row, :]
        inc_sub = inc[start_row:end_row, :]
        kz_sub = kz[start_row:end_row, :]
        mask_sub = mask[start_row:end_row, :]

        if isinstance(mu, np.ndarray):
            mu_sub = mu[start_row:end_row, :]
        else:
            mu_sub = mu

        if isinstance(rngslope, np.ndarray):
            slope_sub = rngslope[start_row:end_row, :]
        else:
            slope_sub = rngslope

        # 调用原版 rvoginv
        return rvoginv(
            gamma_sub, phi_sub, inc_sub, kz_sub,
            ext=ext, tdf=tdf, mu=mu_sub, rngslope=slope_sub,
            mask=mask_sub, limit2pi=limit2pi,
            hv_min=hv_min, hv_max=hv_max, hv_step=hv_step,
            ext_min=ext_min, ext_max=ext_max,
            silent=True   # 子线程不打印
        )

    # ====== 多线程执行 ======
    # 自动按行分块
    chunk_size = rows // parrel_idex

    tasks = []
    with ThreadPoolExecutor(max_workers=parrel_idex) as executor:
        for i in range(parrel_idex):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < parrel_idex - 1 else rows
            tasks.append(executor.submit(worker, start, end))

    # ====== 合并结果 ======
    hv_list, ext_tdf_list, conv_list = [], [], []
    for t in tasks:
        hv_sub, ext_tdf_sub, conv_sub = t.result()
        hv_list.append(hv_sub)
        ext_tdf_list.append(ext_tdf_sub)
        conv_list.append(conv_sub)

    # 行拼接回大矩阵
    hv_final = np.vstack(hv_list)
    ext_tdf_final = np.vstack(ext_tdf_list)
    conv_final = np.vstack(conv_list)

    if not silent:
        print(f"[pdopt_parrel] 完成并行 RVoG 反演，线程数={parrel_idex}")

    return hv_final, ext_tdf_final, conv_final




def pdopt_pixel(tm, om, numph=60, reg=0.0):
    """Phase diversity coherence optimization for a single pixel.

    Same functionality as the pdopt function above, but for a single pixel
    only.  This is the function called when plotting a coherence region.

    Arguments:
        tm (array): The polarimetric covariance (T) matrix of the data,
            with dimensions: [num_pol, num_pol].  Note that in the
            HDF5 file, covariance matrix elements below the diagonal are
            zero-valued, in order to save disk space.  The (j,i) elements
            should therefore calculated from the complex conjugate of the
            (i,j) elements using the kapok.lib.makehermitian() function before
            the matrix is passed to this function.  Note: This should be the
            average matrix of the two tracks forming the baseline, assuming
            polarimetric stationarity.
        om (array): The polarimetric interferometric (Omega) matrix of the
            data, with dimensions [num_pol, num_pol].
        numph (int): The number of phase shifts to calculate coherences for.
            The higher the number, the smaller the spacing of the coherences
            around the coherence region perimeter.  The smaller the number,
            the faster the computation time.  Default: 30.
        reg (float): Regularization factor.  The tm matrix is added to
            the matrix reg*Tr(tm)*I, where Tr(tm) is the trace of tm, and I
            is the identity matrix.  Similarly, the omega matrix is added to
            the matrix reg*Tr(om)*I.  This regularization reduces the spread
            of the coherence region for pixels where the backscatter is
            highly polarization dependent.

    Returns:
        gammamax (complex): the optimized coherence with the max eigenvalue.
        gammamin (complex): the optimized coherence with the min eigenvalue.
        gammaregion (array): Every coherence from the solved eigenvalue
            problems.  These coherences will lie around the edge of the
            coherence region.

    """
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

def rvogfwdvol(hv, ext, inc, kz, rngslope=0.0):
    """RVoG forward model volume coherence.

    For a given set of model parameters, calculate the RVoG model coherence.

    Note that all input arguments must be arrays (even if they are one element
    arrays), so that they can be indexed to check for nan or infinite values
    (due to extreme extinction or forest height values).  All input arguments
    must have the same shape.

    Arguments:
        hv (array): Height of the forest volume, in meters.
        ext (array): Wave extinction within the forest volume, in Np/m.
        inc (array): Incidence angle, in radians.
        kz (array): Interferometric vertical wavenumber, in radians/meter.
        rngslope (array): Range-facing terrain slope angle, in radians.  If not
            specified, flat terrain is assumed.

    Returns:
        gamma: Modelled complex coherence.

    """
    # Calculate the propagation coefficients.
    p1 = 2 * ext * np.cos(rngslope) / np.cos(inc - rngslope)
    p2 = p1 + 1j * kz

    # Check for zero or close to zero hv (or kz) values (e.g., negligible
    # volume decorrelation).
    gammav = kz * hv
    ind_novolume = np.isclose(np.abs(gammav), 0)

    # Check for zero or close to zero extinction values (e.g., uniform
    # vertical structure function).
    gammav = p2 * (np.exp(p1 * hv) - 1)
    ind_zeroext = np.isclose(np.abs(gammav), 0) & ~ind_novolume

    # Check for infinite numerator of the volume coherence equation (e.g.,
    # extremely high extinction value).
    gammav = p1 * (np.exp(p2 * hv) - 1)
    ind_nonfinite = ~np.isfinite(gammav) & ~ind_novolume & ~ind_zeroext

    # The remaining indices are where the standard equation should be valid:
    ind = ~ind_zeroext & ~ind_novolume & ~ind_nonfinite

    if np.any(ind_novolume):
        gammav[ind_novolume] = 1

    if np.any(ind_zeroext):
        gammav[ind_zeroext] = ((np.exp(1j * kz * hv) - 1) / (1j * kz * hv))[ind_zeroext]

    if np.any(ind_nonfinite):
        gammav[ind_nonfinite] = np.exp(1j * hv * kz)[ind_nonfinite]

    if np.any(ind):
        gammav[ind] = ((p1 / p2) * (np.exp(p2 * hv) - 1) / (np.exp(p1 * hv) - 1))[ind]

    return gammav


def rvoginv(gamma, phi, inc, kz, ext=None, tdf=0.7, mu=0.35, rngslope=0.0,
            mask=None, limit2pi=True, hv_min=2, hv_max=45.0, hv_step=0.01,
            ext_min=0.01151, ext_max=0.02875, silent=False):
    """RVoG model inversion.
ext_min=0.01151, ext_max=0.02875
- **gamma (array)**：二维复值数组，包含相干性优化后得到的“高”相干性。
- **phi (array)**：二维复值数组，包含地表相干性（例如通过kapok.topo.groundsolver()计算得到）。
- **inc (array)**：二维数组，包含参考轨道的入射角（单位：弧度）。
- **kz (array)**：二维数组，包含kz值（单位：弧度/米）。
- **ext**：衰减参数的固定值（单位：Neper/米）。如果未指定，函数将优化ext和hv（固定tdf）。默认值：None。
- **tdf**：时间去相关因子的固定值（范围：0到1）。如果未指定，函数将优化tdf和hv。如果ext和tdf都未指定，tdf固定为1。默认值：None。
- **mu**：gamma输入参数对应的地-体散射比固定值。默认值：0。
- **rngslope (array)**：地形在距离方向的坡度角（单位：弧度）。默认值：0（即平坦地形）。
- **mask (array)**：布尔数组。当(mask == True)时，该像素将进行反演；当(mask == False)时，该像素将被忽略，hv设为-1。默认None
- **limit2pi (bool)**：如果为True，函数将不允许hv超过2π/kz（由kz值确定的高度模糊性）。如果为False，则无此限制。默认值：True。
- **hv_min (float or array)**：允许的最小森林高度（单位：米）。默认值：0。
- **hv_max (float or array)**：允许的最大森林高度（单位：米）。默认值：50。
- **hv_step (float)**：函数将在多轮搜索中逐步缩小搜索步长，直到步长小于hv_step。默认值：0.01米。
- **ext_min (float)**：最小衰减值（单位：Np/m）。默认值：0.00115 Np/m（约0.01 dB/m）。
- **ext_max (float)**：最大衰减值（单位：Np/m）。默认值：0.115 Np/m（约1 dB/m）。
- **silent (bool)**：如果设为True，则不显示状态更新。默认值：False。

        Returns:
            hvmap (array): Array of inverted forest height values, in meters.
            extmap/tdfmap (array): If ext was specified, array of inverted tdf
                values will be returned here.  If tdf was specified, array
                of inverted ext values will be returned.
            converged (array): A 2D boolean array.  For each pixel, if
                |observed gamma - modelled gamma| <= 0.01, that pixel is
                marked as converged.  Otherwise, converged will be False for
                that pixel.  Pixels where converged == False suggest that the
                RVoG model could not find a good fit for that pixel, and the
                parameter estimates may be invalid.

    """
    if not silent:
        print('kapok.rvog.rvoginv | Beginning RVoG model inversion. (' + time.ctime() + ')')
    dim = np.shape(gamma)

    if mask is None:
        mask = np.ones(dim, dtype='bool')

    if np.all(limit2pi) or (limit2pi is None):
        limit2pi = np.ones(dim, dtype='bool')
    elif np.all(limit2pi == False):
        limit2pi = np.zeros(dim, dtype='bool')

    if isinstance(hv_max, (collections.abc.Sequence, np.ndarray)):
        hv_max_clip = hv_max.copy()[mask]
        hv_max = np.nanmax(hv_max)
    else:
        hv_max_clip = None

    if isinstance(hv_min, (collections.abc.Sequence, np.ndarray)):
        hv_min_clip = hv_min.copy()[mask]
        hv_min = np.nanmin(hv_min)
    else:
        hv_min_clip = None

    hv_samples = int((hv_max - hv_min) * 2 + 1)  # Initial Number of hv Bins in Search Grid
    hv_vector = np.linspace(hv_min, hv_max, num=hv_samples)

    if tdf is not None:
        ext_samples = 40
        ext_vector = np.linspace(ext_min, ext_max, num=ext_samples)
    elif ext is None:
        tdf = 1.0
        ext_samples = 40
        ext_vector = np.linspace(ext_min, ext_max, num=ext_samples)
    else:
        ext_vector = [-1.0]

    # Use mask to clip input data.
    gammaclip = gamma[mask]
    phiclip = phi[mask]
    incclip = inc[mask]
    kzclip = kz[mask]
    limit2piclip = limit2pi[mask]

    if isinstance(mu, (collections.abc.Sequence, np.ndarray)):
        muclip = mu[mask]
    elif isinstance(mu, dict):
        if not silent:
            print('kapok.rvog.rvoginv | Using LUT for mu as a function of forest height.')
        muclip = None
    else:
        muclip = np.ones(gammaclip.shape, dtype='float32') * mu

    if isinstance(rngslope, (collections.abc.Sequence, np.ndarray)):
        rngslopeclip = rngslope[mask]
    else:
        rngslopeclip = np.ones(gammaclip.shape, dtype='float32') * rngslope

    if isinstance(ext, (collections.abc.Sequence, np.ndarray)):
        extclip = ext[mask]
    elif isinstance(ext, dict):
        if not silent:
            print('kapok.rvog.rvoginv | Using LUT for extinction as a function of forest height.')
        extclip = None
    elif ext is not None:
        extclip = np.ones(gammaclip.shape, dtype='float32') * ext
    elif isinstance(tdf, (collections.abc.Sequence, np.ndarray)):
        tdfclip = tdf[mask]
    elif isinstance(tdf, dict):
        if not silent:
            print('kapok.rvog.rvoginv | Using LUT for temporal decorrelation magnitude as a function of forest height.')
        tdfclip = None
    elif tdf is not None:
        tdfclip = np.ones(gammaclip.shape, dtype='float32') * tdf

    # Arrays to store the fitted parameters:
    hvfit = np.zeros(gammaclip.shape, dtype='float32')

    if ext is None:
        extfit = np.zeros(gammaclip.shape, dtype='float32')
        if not silent:
            print('kapok.rvog.rvoginv | Solving for forest height and extinction, with fixed temporal decorrelation.')
    else:
        tdffit = np.zeros(gammaclip.shape, dtype='float32')
        if not silent:
            print(
                'kapok.rvog.rvoginv | Solving for forest height and temporal decorrelation magnitude, with fixed extinction.')

    # Variables for optimization:
    mindist = np.ones(gammaclip.shape, dtype='float32') * 1e9
    convergedclip = np.ones(gammaclip.shape, dtype='bool')
    threshold = 0.01  # threshold for convergence

    if not silent:
        print(
            'kapok.rvog.rvoginv | Performing repeated searches over smaller parameter ranges until hv step size is less than ' + str(
                hv_step) + ' m.')
        print('kapok.rvog.rvoginv | Beginning pass #1 with hv step size: ' + str(
            np.round(hv_vector[1] - hv_vector[0], decimals=3)) + ' m. (' + time.ctime() + ')')

    for n, hv_val in enumerate(hv_vector):
        if not silent:
            print('kapok.rvog.rvoginv | Progress: ' + str(
                np.round(n / hv_vector.shape[0] * 100, decimals=2)) + '%. (' + time.ctime() + ')     ', end='\r')
        for ext_val in ext_vector:
            if isinstance(mu, dict):
                muclip = np.interp(hv_val, mu['x'], mu['y'])

            if ext is None:
                if isinstance(tdf, dict):
                    tdfclip = np.interp(hv_val, tdf['x'], tdf['y'])

                gammav_model = rvogfwdvol(hv_val, ext_val, incclip, kzclip, rngslope=rngslopeclip)
                gamma_model = phiclip * (muclip + tdfclip * gammav_model) / (muclip + 1)
                dist = np.abs(gammaclip - gamma_model)
            else:
                if isinstance(ext, dict):
                    extclip = np.interp(hv_val, ext['x'], ext['y'])

                gammav_model = rvogfwdvol(hv_val, extclip, incclip, kzclip, rngslope=rngslopeclip)
                tdf_val = np.abs((gammaclip * (muclip + 1) - phiclip * muclip) / (phiclip * gammav_model))
                gamma_model = phiclip * (muclip + tdf_val * gammav_model) / (muclip + 1)
                dist = np.abs(gammaclip - gamma_model)

            # If potential vegetation height is greater than
            # 2*pi ambiguity height, and the limit2pi option
            # is set to True, remove these as potential solutions:
            ind_limit = limit2piclip & (hv_val > np.abs(2 * np.pi / kzclip))
            if np.any(ind_limit):
                dist[ind_limit] = 1e10

            # If hv_min and hv_max were set to arrays,
            # ensure that solutions outside of the bounds are excluded.
            if hv_min_clip is not None:
                ind_limit = (hv_val < hv_min_clip)
                if np.any(ind_limit):
                    dist[ind_limit] = 1e10

            if hv_max_clip is not None:
                ind_limit = (hv_val > hv_max_clip)
                if np.any(ind_limit):
                    dist[ind_limit] = 1e10

            # Best solution so far?
            ind = dist < mindist

            # Then update:
            if np.any(ind):
                mindist[ind] = dist[ind]
                hvfit[ind] = hv_val
                if ext is None:
                    extfit[ind] = ext_val
                else:
                    tdffit[ind] = tdf_val[ind]

    hv_inc = hv_vector[1] - hv_vector[0]
    if ext is None:
        ext_inc = ext_vector[1] - ext_vector[0]
    else:
        ext_inc = 1e-10

    itnum = 1
    while (hv_inc > hv_step):
        itnum += 1
        hv_low = hvfit - hv_inc
        hv_high = hvfit + hv_inc
        hv_val = hv_low.copy()
        hv_inc /= 10

        if ext is None:
            ext_low = extfit - ext_inc
            ext_low[ext_low < ext_min] = ext_min
            ext_high = extfit + ext_inc
            ext_high[ext_high > ext_max] = ext_max
            ext_val = ext_low.copy()
            ext_inc /= 10
        else:
            ext_low = np.array(ext_min, dtype='float32')
            ext_high = np.array(ext_max, dtype='float32')
            ext_val = ext_low.copy()
            ext_inc = 10.0

        if not silent:
            print('kapok.rvog.rvoginv | Beginning pass #' + str(itnum) + ' with hv step size: ' + str(
                np.round(hv_inc, decimals=3)) + ' m. (' + time.ctime() + ')')
        while np.all(hv_val < hv_high):
            if not silent:
                print('kapok.rvog.rvoginv | Progress: ' + str(
                    np.round((hv_val - hv_low) / (hv_high - hv_low) * 100, decimals=2)[
                        0]) + '%. (' + time.ctime() + ')     ', end='\r')

            while np.all(ext_val < ext_high):
                if isinstance(mu, dict):
                    muclip = np.interp(hv_val, mu['x'], mu['y'])

                if ext is None:
                    if isinstance(tdf, dict):
                        tdfclip = np.interp(hv_val, tdf['x'], tdf['y'])

                    gammav_model = rvogfwdvol(hv_val, ext_val, incclip, kzclip, rngslope=rngslopeclip)
                    gamma_model = phiclip * (muclip + tdfclip * gammav_model) / (muclip + 1)
                    dist = np.abs(gammaclip - gamma_model)
                else:
                    if isinstance(ext, dict):
                        extclip = np.interp(hv_val, ext['x'], ext['y'])

                    gammav_model = rvogfwdvol(hv_val, extclip, incclip, kzclip, rngslope=rngslopeclip)
                    tdf_val = np.abs((gammaclip * (muclip + 1) - phiclip * muclip) / (phiclip * gammav_model))
                    gamma_model = phiclip * (muclip + tdf_val * gammav_model) / (muclip + 1)
                    dist = np.abs(gammaclip - gamma_model)

                # If potential vegetation height is greater than
                # 2*pi ambiguity height, and the limit2pi option
                # is set to True, remove these as potential solutions:
                ind_limit = limit2piclip & (hv_val > np.abs(2 * np.pi / kzclip))
                if np.any(ind_limit):
                    dist[ind_limit] = 1e10

                # If hv_min and hv_max were set to arrays,
                # ensure that solutions outside of the bounds are excluded.
                if hv_min_clip is not None:
                    ind_limit = (hv_val < hv_min_clip)
                    if np.any(ind_limit):
                        dist[ind_limit] = 1e10

                if hv_max_clip is not None:
                    ind_limit = (hv_val > hv_max_clip)
                    if np.any(ind_limit):
                        dist[ind_limit] = 1e10

                # Best solution so far?
                ind = np.less(dist, mindist)

                # Then update:
                if np.any(ind):
                    mindist[ind] = dist[ind]
                    hvfit[ind] = hv_val[ind]
                    if ext is None:
                        extfit[ind] = ext_val[ind]
                    else:
                        tdffit[ind] = tdf_val[ind]

                # Increment the extinction:
                ext_val += ext_inc

            # Increment the forest height:
            hv_val += hv_inc
            ext_val = ext_low.copy()

    # Check convergence rate.
    ind = np.less(mindist, threshold)
    convergedclip[ind] = True
    num_converged = np.sum(convergedclip)
    num_total = len(convergedclip)
    rate = np.round(num_converged / num_total * 100, decimals=2)

    if not silent:
        print('kapok.rvog.rvoginv | Completed.  Convergence Rate: ' + str(rate) + '%. (' + time.ctime() + ')')

    # Rebuild masked arrays into original image size.
    hvmap = np.ones(dim, dtype='float32') * -1
    hvmap[mask] = hvfit

    converged = np.ones(dim, dtype='float32') * -1
    converged[mask] = convergedclip

    if ext is None:
        extmap = np.ones(dim, dtype='float32') * -1
        extmap[mask] = extfit
        return hvmap, extmap, converged
    else:
        tdfmap = np.ones(dim, dtype='float32') * -1
        tdfmap[mask] = tdffit
        return hvmap, tdfmap, converged
