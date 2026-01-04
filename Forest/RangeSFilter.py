# # -*- coding: utf-8 -*-
# """
# Modified on 2025-05-15
# Refactored by ChatGPT
#
def process_all_pols(
    array_HH_master, array_VV_master, array_HV_master, array_VH_master,
    array_HH_slave, array_VV_slave, array_HV_slave, array_VH_slave,
    dem_array, inc_array,
    wavelength=0.234,
    baseline=250.0,
    pixel_spacing=10.0,
    patch_size=128,
    beta_base=2.4,
    beta_scale=10.0
):
    """
    对四极化通道 (HH, VV, HV, VH) 主辅图像进行谱域滤波并返回修改后的 slave 通道。
    所有输入为 NumPy 数组。
    """
    from numpy import pi, gradient, sin, deg2rad
    from scipy.fft import fft, ifft, fftshift, ifftshift
    import numpy as np

    def compute_topo_phase_and_slope(DEM, inc_rad, baseline, wavelength, pixel_spacing):
        slope = np.gradient(DEM, pixel_spacing, axis=1)
        f_DEM = (4 * pi * baseline / (wavelength * np.sin(inc_rad))) * slope * pixel_spacing
        return f_DEM, slope

    def demodulate(S1, S2, f_DEM):
        return S1 * np.exp(-1j * f_DEM / 2), S2 * np.exp(+1j * f_DEM / 2)

    def remodulate(S1f, S2f, f_DEM):
        return S1f * np.exp(+1j * f_DEM / 2), S2f * np.exp(-1j * f_DEM / 2)

    def slope_adaptive_beta(slope, beta_base, scale):
        beta = beta_base + scale * np.abs(slope)
        return np.clip(beta, 1.0, 8.0)

    def spectral_filter(S1_0, S2_0, slope, patch_size, beta_base, beta_scale):
        height, width = S1_0.shape
        S1_out = np.zeros_like(S1_0, dtype=complex)
        S2_out = np.zeros_like(S2_0, dtype=complex)

        for i in range(0, width, patch_size):
            i_end = min(i + patch_size, width)
            local_slope = np.mean(np.abs(slope[:, i:i_end]))
            beta = slope_adaptive_beta(local_slope, beta_base, beta_scale)
            window = np.kaiser(patch_size, beta)
            window = window / np.max(window)

            for row in range(height):
                patch_len = i_end - i
                s1_patch = np.zeros(patch_size, dtype=complex)
                s2_patch = np.zeros(patch_size, dtype=complex)
                s1_patch[:patch_len] = S1_0[row, i:i_end]
                s2_patch[:patch_len] = S2_0[row, i:i_end]

                F1 = fftshift(fft(s1_patch))
                F2 = fftshift(fft(s2_patch))
                F1_filtered = F1 * window
                F2_filtered = F2 * window
                s1f = ifft(ifftshift(F1_filtered))
                s2f = ifft(ifftshift(F2_filtered))

                S1_out[row, i:i_end] = s1f[:patch_len]
                S2_out[row, i:i_end] = s2f[:patch_len]

        return S1_out, S2_out

    # 获取平均入射角（单位：弧度）
    inc_rad = np.deg2rad(np.mean(inc_array))

    # 计算 topographic phase & slope
    f_DEM, slope = compute_topo_phase_and_slope(dem_array, inc_rad, baseline, wavelength, pixel_spacing)

    # 极化通道字典
    master_dict = {
        "HH": array_HH_master, "VV": array_VV_master,
        "HV": array_HV_master, "VH": array_VH_master
    }
    slave_dict = {
        "HH": array_HH_slave, "VV": array_VV_slave,
        "HV": array_HV_slave, "VH": array_VH_slave
    }

    # 输出（修改后的 slave）
    output_slave = {}

    for pol in ["HH", "VV", "HV", "VH"]:
        S1 = master_dict[pol]
        S2 = slave_dict[pol]
        S1_0, S2_0 = demodulate(S1, S2, f_DEM)
        S1f, S2f = spectral_filter(S1_0, S2_0, slope, patch_size, beta_base, beta_scale)
        _, S2_out = remodulate(S1f, S2f, f_DEM)
        output_slave[pol] = S2_out

    return (
        output_slave["HH"],
        output_slave["VV"],
        output_slave["HV"],
        output_slave["VH"]
    )

# Function:
#     Provide a reusable spectral filtering function for PolInSAR preprocessing.
#     Input: SLC arrays, DEM array, incidence angle array, and parameters.
#     Output: Filtered complex SLC arrays.
# """
#
# import numpy as np
# from scipy.fft import fft, ifft, fftshift, ifftshift
#
#
# def slope_adaptive_beta(slope, beta_base=2.4, scale=10.0):
#     beta = beta_base + scale * np.abs(slope)
#     return np.clip(beta, 1.0, 8.0)
#
#
# def compute_topographic_phase_and_slope(DEM, wavelength, baseline, inc_angle_rad, pixel_spacing):
#     slope = np.gradient(DEM, pixel_spacing, axis=1)
#     f_DEM = (4 * np.pi * baseline / (wavelength * np.sin(inc_angle_rad))) * slope * pixel_spacing
#     return f_DEM, slope
#
#
# def demodulate(S1, S2, f_DEM):
#     S1_0 = S1 * np.exp(-1j * f_DEM / 2)
#     S2_0 = S2 * np.exp(+1j * f_DEM / 2)
#     return S1_0, S2_0
#
#
# def remodulate(S1f, S2f, f_DEM):
#     S1 = S1f * np.exp(+1j * f_DEM / 2)
#     S2 = S2f * np.exp(-1j * f_DEM / 2)
#     return S1, S2
#
#
# def spectral_filter_kaiser_adaptive(S1_0, S2_0, slope, patch_size=128, beta_base=2.4, beta_scale=10.0):
#     height, width = S1_0.shape
#     S1_out = np.zeros_like(S1_0, dtype=complex)
#     S2_out = np.zeros_like(S2_0, dtype=complex)
#
#     for i in range(0, width, patch_size):
#         i_end = min(i + patch_size, width)
#         local_slope = np.mean(np.abs(slope[:, i:i_end]))
#         beta = slope_adaptive_beta(local_slope, beta_base, beta_scale)
#         window = np.kaiser(patch_size, beta)
#         window = window / np.max(window)
#
#         for row in range(height):
#             patch_len = i_end - i
#             s1_patch = np.zeros(patch_size, dtype=complex)
#             s2_patch = np.zeros(patch_size, dtype=complex)
#             s1_patch[:patch_len] = S1_0[row, i:i_end]
#             s2_patch[:patch_len] = S2_0[row, i:i_end]
#
#             F1 = fftshift(fft(s1_patch))
#             F2 = fftshift(fft(s2_patch))
#
#             F1_filtered = F1 * window
#             F2_filtered = F2 * window
#
#             s1f = ifft(ifftshift(F1_filtered))
#             s2f = ifft(ifftshift(F2_filtered))
#
#             S1_out[row, i:i_end] = s1f[:patch_len]
#             S2_out[row, i:i_end] = s2f[:patch_len]
#
#     return S1_out, S2_out
#
#
# def slope_kaiser_filter(S1, S2, DEM, inc_angle_rad, wavelength, baseline, pixel_spacing=15.4854536679144860755741319736, patch_size=128):
#     """
#     Main filtering interface for external usage.
#
#     Parameters:
#         S1, S2         : np.ndarray, complex64, SLC master/slave data.
#         DEM            : np.ndarray, float32, DEM array (same size as SLC).
#         inc_angle_rad  : np.ndarray or float, radar incidence angle in radians (either scalar or 2D).
#         wavelength     : float, radar wavelength in meters.
#         baseline       : float, perpendicular baseline in meters.
#         pixel_spacing  : float, pixel spacing in range direction in meters.
#         patch_size     : int, spectral window size.
#
#     Returns:
#         S1_filtered, S2_filtered : filtered complex SLC arrays.
#         slope                    : slope used in filtering.
#     """
#     if isinstance(inc_angle_rad, np.ndarray):
#         assert inc_angle_rad.shape == DEM.shape, "If incidence angle is array, must match DEM size."
#     else:
#         inc_angle_rad = np.full_like(DEM, inc_angle_rad, dtype=np.float32)
#
#     f_DEM, slope = compute_topographic_phase_and_slope(DEM, wavelength, baseline, inc_angle_rad, pixel_spacing)
#     S1_0, S2_0 = demodulate(S1, S2, f_DEM)
#     S1_f, S2_f = spectral_filter_kaiser_adaptive(S1_0, S2_0, slope, patch_size=patch_size)
#     S1_filtered, S2_filtered = remodulate(S1_f, S2_f, f_DEM)
#     return S1_filtered, S2_filtered, slope
