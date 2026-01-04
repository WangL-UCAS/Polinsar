import numpy as np
import matplotlib.pyplot as plt
from Forest import Rovg_not_parrel, make_ground

def rvogregion_ext_sweep(tm, om, kz, inc, ground, reg=0.0):
    """在ext范围内迭代，画出不同ext对应的RVoG模型曲线"""

    inc = np.radians(inc)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.3)

    # 画相干幅度的参考圆圈
    for r in [1.0, 0.75, 0.5, 0.25]:
        circ = plt.Circle((0, 0), r, color='k', fill=False, linestyle='dashed')
        ax.add_artist(circ)

    # 得到 gamma region, 高低相干
    gammahigh, gammalow, gammaall = Rovg_not_parrel.pdopt_pixel(tm, om, reg=reg)

    # Ground coherence estimation（线性拟合）
    gammatemp = np.zeros((2, 2, 2), dtype='complex')
    gammatemp[0, :, :] = gammahigh
    gammatemp[1, :, :] = gammalow
    ground_fit, groundalt, volindex = make_ground.groundsolver(gammatemp, kz=kz, returnall=True, silent=True)

    if volindex[0, 0]:  # 如果需要调换 high/low coherence
        gammahigh, gammalow = gammalow, gammahigh

    ground_fit = ground_fit[0, 0]
    groundalt = groundalt[0, 0]

    # 画区域和线性拟合线
    ax.plot(np.real(gammaall), np.imag(gammaall), '-', linewidth=3, label='Region')
    ax.plot([np.real(ground_fit), np.real(gammalow), np.real(gammahigh)],
            [np.imag(ground_fit), np.imag(gammalow), np.imag(gammahigh)],
            '--g', linewidth=3, label='Line Fit')

    # 画点
    ax.plot(np.real(gammahigh), np.imag(gammahigh), '.', color='DarkGreen', markersize=20, label='Opt. High')
    ax.plot(np.real(gammalow), np.imag(gammalow), '.', color='Maroon', markersize=20, label='Opt. Low')
    ax.plot(np.real(ground_fit), np.imag(ground_fit), '.k', markersize=20, label='Ground')
    ax.plot(np.real(groundalt), np.imag(groundalt), '.', color='orange', markersize=20, label='Alt. Ground')

    # 固定参数
    hv = 30.0
    hv_vector = np.linspace(1, hv, num=int(hv))
    mu_high = 0.5
    mu_low = 0.5
    alpha = 0.9
    nptodb = 20 / np.log(10)
    cos_inc = np.cos(inc)

    # ext取值范围和步长
    ext_values = np.arange(0.01, 1.01, 0.2)

    for ext in ext_values:
        p1 = 2 * ext / nptodb / cos_inc
        p2 = p1 + 1j * kz
        gammav = (p1 / p2) * (np.exp(p2 * hv_vector) - 1) / (np.exp(p1 * hv_vector) - 1)
        gammahigh_model = ground * (mu_high + alpha * gammav) / (mu_high + 1)
        gammalow_model = ground * (mu_low + alpha * gammav[-1]) / (mu_low + 1)
        rvoglocus = np.array([gammahigh_model[-1], gammalow_model])

        # 画高相干体积散射曲线
        ax.plot(np.real(gammahigh_model), np.imag(gammahigh_model), '.', linewidth=1, color='YellowGreen')
        # 画对应的点连接线（高-低）


    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('RVoG Coherence Region with Extinction Sweep')

    # 图例放右上角，避免遮挡
    ax.legend(loc='upper right', fontsize='small', ncol=2)

    plt.show()
