
import numpy as np
import collections
import time
import multiprocessing as mp



import numpy as np


def rvogfwdvol(hv, ext, inc, kz, rngslope):
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

def rvoginv(gamma, phi, inc, kz, rngslope,ext, tdf, mu,
            mask, limit2pi, hv_min, hv_max, hv_step,
            ext_min, ext_max, silent):
    """
   RVoG模型反演。

计算RVoG模型参数，使其生成的模型相干性与一组观测相干性最接近。该模型采用实值体积时间去相关因子（tdf），其物理参数包括森林高度（hv）、雷达波在森林冠层内的衰减（ext），以及地表相干性（phi），其中arg(phi)等于地形相位。此外，地-体散射幅度比（mu）随极化方式变化。

在单基线情况下，为了减少未知参数的数量并确保模型具有唯一解，我们假设高相干性（通过相干性优化获得的相干性）对应的mu是固定的。默认情况下，mu设为零，即假设高相干性没有地面散射分量。因此，我们必须固定衰减值（ext）或时间去相关因子（tdf）。

因此，该函数需要提供ext或tdf参数之一。函数将优化未提供的这两个参数之一，以及森林高度参数。如果两个参数都未提供，则tdf将固定为1.0（即无时间去相关）。

需要注意的是，ext、tdf和mu参数可以作为固定值提供（例如mu=0），也可以作为与gamma尺寸相同的数组提供，或者作为森林高度参数的查找表（LUT）提供。在LUT情况下，dict['x']包含每个LUT分箱的森林高度值，dict['y']包含参数值。函数将使用numpy.interp对森林高度值进行插值。

此外，该函数不能同时固定ext和tdf，函数总会尝试求解其中之一。

### 参数：
- **gamma (array)**：二维复值数组，包含相干性优化后得到的“高”相干性。
- **phi (array)**：二维复值数组，包含地表相干性（例如通过kapok.topo.groundsolver()计算得到）。
- **inc (array)**：二维数组，包含参考轨道的入射角（单位：弧度）。
- **kz (array)**：二维数组，包含kz值（单位：弧度/米）。
- **ext**：衰减参数的固定值（单位：Neper/米）。如果未指定，函数将优化ext和hv（固定tdf）。默认值：None。
- **tdf**：时间去相关因子的固定值（范围：0到1）。如果未指定，函数将优化tdf和hv。如果ext和tdf都未指定，tdf固定为1。默认值：None。
- **mu**：gamma输入参数对应的地-体散射比固定值。默认值：0。
- **rngslope (array)**：地形在距离方向的坡度角（单位：弧度）。默认值：0（即平坦地形）。
- **mask (array)**：布尔数组。当(mask == True)时，该像素将进行反演；当(mask == False)时，该像素将被忽略，hv设为-1。
- **limit2pi (bool)**：如果为True，函数将不允许hv超过2π/kz（由kz值确定的高度模糊性）。如果为False，则无此限制。默认值：True。
- **hv_min (float or array)**：允许的最小森林高度（单位：米）。默认值：0。
- **hv_max (float or array)**：允许的最大森林高度（单位：米）。默认值：50。
- **hv_step (float)**：函数将在多轮搜索中逐步缩小搜索步长，直到步长小于hv_step。默认值：0.01米。
- **ext_min (float)**：最小衰减值（单位：Np/m）。默认值：0.00115 Np/m（约0.01 dB/m）。
- **ext_max (float)**：最大衰减值（单位：Np/m）。默认值：0.115 Np/m（约1 dB/m）。
- **silent (bool)**：如果设为True，则不显示状态更新。默认值：False。

### 返回值：
- **hvmap (array)**：反演得到的森林高度数组（单位：米）。
- **extmap/tdfmap (array)**：如果指定了ext，则返回tdf的反演值数组；如果指定了tdf，则返回ext的反演值数组。
- **converged (array)**：二维布尔数组。如果|观测gamma - 模型gamma| ≤ 0.01，则该像素被标记为收敛（True）；否则为False。如果某像素converged == False，则表明RVoG模型未能找到该像素的良好拟合解，参数估计可能无效。

    :param gamma:
    :param phi:
    :param inc:
    :param kz:
    :param ext:
    :param tdf:
    :param mu:
    :param rngslope:
    :param mask:
    :param limit2pi:
    :param hv_min:
    :param hv_max:
    :param hv_step:
    :param ext_min:
    :param ext_max:
    :param silent:
    :return:
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

                gammav_model = rvogfwdvol(hv_val, ext_val, incclip, kzclip, rngslopeclip)
                gamma_model = phiclip * (muclip + tdfclip * gammav_model) / (muclip + 1)
                dist = np.abs(gammaclip - gamma_model)
            else:
                if isinstance(ext, dict):
                    extclip = np.interp(hv_val, ext['x'], ext['y'])

                gammav_model = rvogfwdvol(hv_val, extclip, incclip, kzclip, rngslopeclip)
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

                    gammav_model = rvogfwdvol(hv_val, ext_val, incclip, kzclip, rngslopeclip)
                    gamma_model = phiclip * (muclip + tdfclip * gammav_model) / (muclip + 1)
                    dist = np.abs(gammaclip - gamma_model)
                else:
                    if isinstance(ext, dict):
                        extclip = np.interp(hv_val, ext['x'], ext['y'])

                    gammav_model = rvogfwdvol(hv_val, extclip, incclip, kzclip, rngslopeclip)
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

def worker_function(gamma_part, phi_part, inc_part, kz_part, rngslope_part, ext_part, tdf_part, mu_part,
                     mask_part, limit2pi_part, hv_min_part, hv_max_part,
                    hv_step, ext_min, ext_max, silent, result_dict, idx):
    """工作函数，用于处理单个数据块的反演

    """
    # 调用rvoginv处理当前数据块
    results = rvoginv(
        gamma=gamma_part,
        phi=phi_part,
        inc=inc_part,
        kz=kz_part,
        rngslope=rngslope_part,
        ext=ext_part,
        tdf=tdf_part,
        mu=mu_part,
        mask=mask_part,
        limit2pi=limit2pi_part,
        hv_min=hv_min_part,
        hv_max=hv_max_part,
        hv_step=hv_step,
        ext_min=ext_min,
        ext_max=ext_max,
        silent=silent
    )
    result_dict[idx] = results

# #def rvoginv(gamma, phi, inc, kz, rngslope,ext=None, tdf=0.7, mu=0.1,
#             mask=None, limit2pi=True, hv_min=0.0, hv_max=50.0, hv_step=0.01,
#             ext_min=0.00115, ext_max=0.115, silent=False):
# def rvoginv(gamma, phi, inc, kz, rngslope,ext, tdf=, mu=,
#             mask, limit2pi, hv_min, hv_max, hv_step,
#             ext_min, ext_max, silent):

def Rvog(gamma, phi, inc, kz,rngslope,ext=None, tdf=0.8, mu=0.1,
         mask=None, limit2pi=True, hv_min=0, hv_max=55.0, hv_step=0.01,
         ext_min=0.00115, ext_max=0.115, silent=False, num_processes=10):
    """多进程并行处理RVoG模型反演
        ### 参数：
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

    """
    if num_processes < 2:
        # 单进程直接调用rvoginv
        return rvoginv(gamma, phi, inc, kz,rngslope, ext=ext, tdf=tdf, mu=mu,
                       mask=mask, limit2pi=limit2pi, hv_min=hv_min, hv_max=hv_max,
                       hv_step=hv_step, ext_min=ext_min, ext_max=ext_max, silent=silent)

    # 辅助函数：根据参数类型分割数据
    def split_data(data, num_p, axis=0):
        if isinstance(data, np.ndarray) and data.ndim >= 1:
            return np.array_split(data, num_p, axis=axis)
        else:
            return [data] * num_p  # 非数组参数每个进程复制一份

    # 需要分割的参数及其对应数据
    params_to_split = {
        'gamma': gamma,
        'phi': phi,
        'inc': inc,
        'kz': kz,
        'ext': ext,
        'tdf': tdf,
        'mu': mu,
        'rngslope': rngslope,
        'mask': mask,
        'limit2pi': limit2pi,
        'hv_min': hv_min,
        'hv_max': hv_max,
    }

    # 分割所有参数
    split_params = {}
    for key in params_to_split:
        split_params[key] = split_data(params_to_split[key], num_processes, axis=0)

    # 其他固定参数
    other_args = {
        'hv_step': hv_step,
        'ext_min': ext_min,
        'ext_max': ext_max,
        'silent': silent,
    }

    # 创建进程间共享的结果字典
    manager = mp.Manager()
    result_dict = manager.dict()

    # 创建并启动进程
    processes = []
    for i in range(num_processes):
        # 提取当前进程的参数块
        args = {
            'gamma_part': split_params['gamma'][i],
            'phi_part': split_params['phi'][i],
            'inc_part': split_params['inc'][i],
            'kz_part': split_params['kz'][i],
            'ext_part': split_params['ext'][i],
            'tdf_part': split_params['tdf'][i],
            'mu_part': split_params['mu'][i],
            'rngslope_part': split_params['rngslope'][i],
            'mask_part': split_params['mask'][i],
            'limit2pi_part': split_params['limit2pi'][i],
            'hv_min_part': split_params['hv_min'][i],
            'hv_max_part': split_params['hv_max'][i],
        }
        args.update(other_args)
        args['result_dict'] = result_dict
        args['idx'] = i

        p = mp.Process(target=worker_function, kwargs=args)
        processes.append(p)
        p.start()  # 启动进程

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 按顺序收集结果
    results = [result_dict[i] for i in range(num_processes)]
    # 合并各块结果
    hvmap = np.vstack([r[0] for r in results])
    param_map = np.vstack([r[1] for r in results])
    converged = np.vstack([r[2] for r in results])

    # 根据输入参数确定返回的是ext还是tdf
    if ext is None:
        return hvmap, param_map, converged
    else:
        return hvmap, param_map, converged