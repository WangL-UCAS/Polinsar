import numpy as np
import os
from numpy.fft import fft2, fftshift
import sys
import argparse

from osgeo import gdal


import numpy as np
import os
import math

def flat_earth_estimation(master_data, slave_data, nwr, nwc, output_dir, output_format='realdeg'):
    """
    参数说明：
    - master_data: np.ndarray，shape=(nr, nc)，复数主影像
    - slave_data: np.ndarray，shape=(nr, nc)，复数辅影像
    - nwr, nwc: 滑窗尺寸（如：64x64）
    - output_dir: 输出目录
    - output_format: 可选 ["realrad", "realdeg", "cmplx"]
    """

    assert master_data.shape == slave_data.shape, "主影像和辅影像大小不一致"
    nr, nc = master_data.shape

    pi = np.pi
    nfft_lig = int(2**math.ceil(math.log2(4 * nwr)))
    nfft_col = int(2**math.ceil(math.log2(4 * nwc)))
    nfft_ligs2 = nfft_lig // 2
    nfft_cols2 = nfft_col // 2

    interf = np.zeros((nfft_lig, nfft_col), dtype=np.complex64)

    # 计算主辅影像干涉图（子窗口）
    master_win = master_data[:nwr, :nwc]
    slave_win = slave_data[:nwr, :nwc]
    interf[:nwr, :nwc] = master_win * np.conj(slave_win)

    # 进行二维FFT
    fft2d = np.fft.fftshift(np.fft.fft2(interf, s=(nfft_lig, nfft_col)))
    power = np.abs(fft2d) ** 2
    lig_max, col_max = np.unravel_index(np.argmax(power), power.shape)

    # 计算最大频率偏移（平地相位斜率）
    iimax = lig_max / nfft_lig
    if lig_max > nfft_ligs2:
        iimax = (lig_max - nfft_lig) / nfft_lig

    jjmax = col_max / nfft_col
    if col_max > nfft_cols2:
        jjmax = (col_max - nfft_col) / nfft_col

    # 构建整幅图像的平地相位项
    px = np.exp(1j * 2 * pi * iimax * np.arange(nr)).reshape((-1, 1))  # shape=(nr,1)
    py = np.exp(1j * 2 * pi * jjmax * np.arange(nc)).reshape((1, -1))  # shape=(1,nc)
    flat_phase = px @ py  # shape=(nr,nc)

    # 输出处理
    output_file = os.path.join(output_dir, "flat_earth_fft.bin")
    flat_phase_output = None

    if output_format == "cmplx":
        flat_phase_output = flat_phase.astype(np.complex64)
    else:
        phase = np.angle(flat_phase).astype(np.float32)
        if output_format == "realdeg":
            phase = np.degrees(phase)
        flat_phase_output = phase

    with open(output_file, "wb") as f:
        f.write(flat_phase_output.tobytes())

    print(f"输出完成：{output_file}")


if __name__ == "__main__":
    out_file = r'E:\forest\Project\Forest'
    master_cplx = gdal.Open(r'E:\forest\Project\Forest\830_4042\830_HH', gdal.GA_ReadOnly)
    master_cplx = master_cplx.GetRasterBand(1).ReadAsArray()
    slave_cplx = gdal.Open(r'E:\forest\Project\Forest\909_4042\909_HH', gdal.GA_ReadOnly)
    slave_cplx = slave_cplx.GetRasterBand(1).ReadAsArray()

    # flat_earth_estimation(master_cplx, slave_cplx, 100, 100, out_file,output_format='realdeg')




