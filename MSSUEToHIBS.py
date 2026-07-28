import random

import numpy as np
import matplotlib.pyplot as plt

from IMT_Simulation import clutter_loss

# ==================== 常量与参数 ====================
R_earth = 6371.0                     # 地球半径 (km)
c = 3e8                               # 光速 (m/s)
f_MHz = 2010                        # 频率 (MHz)
f_Hz = f_MHz * 1e6                    # 频率 (Hz)
lambda_m = c / f_Hz                    # 波长 (m)

# 噪声参数（假设 HIBS 接收机）
B_Hz = 1e6                             # 接收机带宽 (Hz) 与 MSS 发射带宽一致
NF_dB = 5                              # 噪声系数 (dB)
N0_dBm = -174 + 10 * np.log10(B_Hz) + NF_dB   # 噪声功率 (dBm)
print(f"噪声功率 N = {N0_dBm:.2f} dBm")


# ==================== MSS 系统参数 ====================
# 多个系统保留在字典中，方便切换
mss_systems = {
    'LEO1': {'power_density_dBW_Hz': -35, 'terminal_gain_dBi': -5},
    'LEO2': {'power_density_dBW_Hz': -60, 'terminal_gain_dBi': -5},
    'LEO3': {'power_density_dBW_Hz': -55.6, 'terminal_gain_dBi': -5},
    'LEO4    ': {'power_density_dBW_Hz': -55.6, 'terminal_gain_dBi': -5},
    'MEO3': {'power_density_dBW_Hz': -60, 'terminal_gain_dBi': -5},
    'LEO5': {'power_density_dBW_Hz': -60, 'terminal_gain_dBi': -5}
}

# 选择系统 (可修改)
selected_system = 'LEO3'
mss = mss_systems[selected_system]
# 计算 MSS 发射功率 (dBm)
P_t_dBm = mss['power_density_dBW_Hz'] + 30 + 10 * np.log10(B_Hz)   # 30 dBm
G_tx_dBi = mss['terminal_gain_dBi']

print(f"选用 MSS 系统: {selected_system}")
print(f"MSS 发射功率: {P_t_dBm:.2f} dBm, 终端增益: {G_tx_dBi} dBi")

# HIBS 参数（选择第二层小区 cell_layer=2）
H_km = 20                             # HIBS 高度 (km)
H_m = H_km * 1000

# 第二层小区：4x2 阵列
G_max_dBi = 24                           # 最大增益 (dBi)，由 EIRP 58 dBm - 传导功率34 dBm 得到
beamwidth_h_deg = 25                      # 水平 3dB 波束宽度 (度)
beamwidth_v_deg = 51                      # 垂直 3dB 波束宽度 (度)
tilt_deg = 23                             # 下倾角 (度)
Am_dB = 30                                # 前后比 (dB)
SLAv_dB = 30                              # 垂直旁瓣抑制 (dB)

# 可视距离 (km)
d_vis_km = np.sqrt(2 * R_earth * H_km + H_km**2)
print(f"可视距离 = {d_vis_km:.2f} km")

# 仿真参数
N_snapshots = 10000                        # 每个隔离距离的快照数
density_ms = 0.01                          # MSS 终端密度 (个/km²)
R_sim_km = d_vis_km                             # 仿真区域半径 (km)
area_sim_km2 = np.pi * R_sim_km**2
# N_ms = int(density_ms * area_sim_km2)      # 每个快照的终端数
N_ms = 1
print(f"每个快照 MSS 终端数: {N_ms}")

# 隔离距离列表 (HIBS 地面投影点到仿真区域中心的水平距离, km)
isolation_distances = np.arange(180, 181, 1)  # 从 10 km 到 200 km，步长 20 km
# 如果需要更多点，可以增加步长或使用更密的列表

# ==================== 辅助函数 ====================
def fspl(d_km):
    """自由空间路径损耗 (dB)，d_km 单位 km，f_MHz 全局"""
    return 20 * np.log10(d_km) + 20 * np.log10(f_MHz) + 32.4

def hbs_gain(phi_deg, delta_deg, phi_tilt, delta_tilt, bw_h, bw_v, G_max, Am, SLAv):
    """
    计算 HIBS 天线在给定方向上的增益 (dBi)
    参数：
        phi_deg    : 从 HIBS 看 MSS 的方位角 (度，正北为0，顺时针)
        delta_deg  : 从 HIBS 看 MSS 的俯角 (度，水平为0，向下为正)
        phi_tilt   : 天线主瓣方位角 (度)
        delta_tilt : 天线主瓣下倾角 (度)
        bw_h, bw_v : 水平/垂直 3dB 波束宽度 (度)
        G_max      : 天线最大增益 (dBi)
        Am, SLAv   : 前后比和垂直旁瓣抑制 (dB)
    返回：
        增益 (dBi)
    """
    # 计算相对于主瓣的偏移角
    dphi = phi_deg - phi_tilt
    dphi = (dphi + 180) % 360 - 180          # 归一化到 [-180, 180)
    ddelta = delta_deg - delta_tilt

    # 水平方向图衰减 (dB)
    A_H = -min(12 * (abs(dphi) / bw_h)**2, Am)
    # 垂直方向图衰减 (dB)
    A_V = -min(12 * (abs(ddelta) / bw_v)**2, SLAv)

    # 组合衰减：取 -[A_H+A_V] 与 Am 的较小值
    combined_atten = -(A_H + A_V)            # 正数
    attenuation = min(combined_atten, Am)
    gain = G_max - attenuation
    return gain

# ==================== 主仿真循环 ====================
# 存储每个隔离距离下的干扰结果 (列表，每个元素是 N_snapshots 个干扰值)
results_by_distance = {}

for D_offset_km in isolation_distances:
    print(f"正在计算隔离距离 = {D_offset_km} km ...")
    I_dBm_list = []   # 存储当前距离下所有快照的集总干扰 (dBm)

    for snap in range(N_snapshots):
        # 步骤1: 在仿真区域内随机部署 MSS 终端
        # 极坐标生成圆内均匀分布的点
        u = np.random.uniform(0, 1, N_ms)          # 半径平方均匀分布
        theta = np.random.uniform(0, 2*np.pi, N_ms)
        r = R_sim_km * np.sqrt(u)                  # 半径
        x = D_offset_km + r * np.cos(theta)        # 仿真区域中心在 (D_offset, 0)
        y = r * np.sin(theta)

        I_linear_sum = 0.0                         # 集总干扰线性值 (mW)

        # 遍历每个 MSS 终端
        for i in range(N_ms):
            xi, yi = x[i], y[i]

            # 计算斜距
            d_h_km = np.sqrt(xi**2 + yi**2)        # 水平距离 (km)
            d_km = np.sqrt(d_h_km**2 + H_km**2)    # 斜距 (km)

            # 步骤4: 路径损耗
            PL_dB = fspl(d_km)
            # 计算杂波损耗
            closs = clutter_loss(f_MHz / 1000, random.uniform(0, 1), d_km)
            clossmax = clutter_loss(f_MHz / 1000, random.uniform(0, 1), 2)
            if clossmax < closs:
                closs = clossmax

            # 步骤3: MSS 终端在 HIBS 方向的天线增益 (恒定)
            G_tx_lin = 10**(G_tx_dBi / 10)

            # 步骤5: HIBS 在 MSS 方向的天线增益
            # 计算从 HIBS 看 MSS 的方位角和俯角
            # 方位角：正北为0，顺时针，atan2(x, y)
            phi_ms = np.arctan2(xi, yi) * 180 / np.pi
            # 俯角：水平为0，向下为正，arctan(H / d_h)
            if d_h_km == 0:
                delta_ms = 90.0
            else:
                delta_ms = np.arctan(H_km / d_h_km) * 180 / np.pi

            # HIBS 天线方位角设为0（可根据需要随机化，这里固定）
            azimuth_deg = 0.0
            G_rx_dBi = hbs_gain(phi_ms, delta_ms,
                                azimuth_deg, tilt_deg,
                                beamwidth_h_deg, beamwidth_v_deg,
                                G_max_dBi, Am_dB, SLAv_dB)
            G_rx_lin = 10**(G_rx_dBi / 10)

            # 其他损耗 (假设为0 dB)
            Loss_lin = 1.0

            # 步骤6: 单个 MSS 终端的干扰功率 (线性)
            P_t_lin = 10**(P_t_dBm / 10)
            I_n_lin = P_t_lin * G_tx_lin * G_rx_lin / (10**((PL_dB+closs)/10)) * Loss_lin
            I_linear_sum += I_n_lin

        # 步骤8: 集总干扰 (dBm)
        if I_linear_sum > 0:
            I_total_dBm = 10 * np.log10(I_linear_sum)
        else:
            I_total_dBm = -np.inf
        I_dBm_list.append(I_total_dBm)

    results_by_distance[D_offset_km] = I_dBm_list

# ==================== 绘制 CDF 曲线 ====================
plt.figure(figsize=(8, 6))

# 计算 I/N = -6 dB 对应的干扰门限
I_threshold_dBm = N0_dBm - 6   # -115 dBm

# 为每个隔离距离绘制 CDF
for D, I_list in results_by_distance.items():
    I_sorted = np.sort(I_list)+109-18
    cdf = np.arange(1, len(I_sorted)+1) / len(I_sorted)
    plt.plot(I_sorted, cdf, label=f'D = {D} km')

# 标注门限线
plt.axvline(x=-6, color='red', linestyle='--', linewidth=1,
            label=f'I/N = -6 dB (I = {I_threshold_dBm:.1f} dBm)')

plt.xlabel('I/N')
# plt.xlabel(f"CDF P(X ≤ {I_threshold_dBm})={mark_y}")
plt.ylabel('CDF')
plt.title('CDF of Interference from MSS to HIBS')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("CDFfromMSStoHIBS.jpg")
plt.show()

# ==================== 可选：找出满足条件的隔离距离 ====================
# 例如，要求 90% 的干扰低于门限
target_prob = 0.9
satisfied_distances = []
for D, I_list in results_by_distance.items():
    I_array = np.array(I_list)
    prob_below = np.mean(I_array < I_threshold_dBm)
    if prob_below >= target_prob:
        satisfied_distances.append(D)
        print(f"距离 {D} km: {prob_below*100:.1f}% 的干扰低于门限")

if satisfied_distances:
    print(f"\n满足 {target_prob*100}% 干扰低于门限的最小隔离距离: {min(satisfied_distances)} km")
else:
    print(f"\n所有测试距离均未达到 {target_prob*100}% 低于门限的条件")