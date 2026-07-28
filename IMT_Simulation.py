import math
import os
from math import pi, sin, sqrt, cos, log, acos, atan, log10, radians, atan2, degrees

from fontTools.ttLib.tables.G__l_a_t import Glat_format_0
from scipy.signal import kaiserord
from scipy.stats import norm

from Parm_Simulation import BS_height, BS_ue_height

import numpy as np

# 发射增益，扩展AAS天线模型
def gain_calc(UE_v, UE_h, S_v, S_h):
    # UE_h, UE_v 发射方指向接收方的方位角和仰角
    # S_h, S_v 发射方指向受干扰方的方位角和仰角

    # AAS天线参数
    h_3dB = 90 / 180 * pi
    v_3dB = 65 / 180 * pi
    Am = 30
    # 前后比
    SLV = 30
    Msub = 3

    # 元件增益
    Ge_Max = 6.4
    M = 4
    N = 8

    # Horizontal/Vertical radiating element/sub-array spacing, dh /dv
    dh = 0.5
    dv = 2.1
    subtilt = 3 / 180 * pi
    # Peak normalized element radiation pattern
    A = min(-(-min(12 * ((S_h / h_3dB) ** 2), Am) - min(12 * (((S_v - pi / 2) / v_3dB) ** 2), SLV)), Am)
    # print(A)
    # Peak gain normalized element radiation pattern
    Ae = Ge_Max - A
    # print(Ae)
    # Sub-array excitation
    # Wm = exp(2j * pi * (m - 1) * 0.7 * sin(3)) / sqrt(3)
    # Sub-array radiation pattern
    # Vm = exp(2j * pi * (m - 1) * 0.7 * cos(S_v))
    sumWV = 0
    m = 1
    while m <= Msub:
        Wm = complex(
            cos(pi * (m - 1) * 0.7 * sin(subtilt) * 2),
            sin(pi * (m - 1) * 0.7 * sin(subtilt) * 2)
        )
        Vm = complex(
            cos(2 * pi * (m - 1) * 0.7 * cos(S_v)),
            sin(2 * pi * (m - 1) * 0.7 * cos(S_v))
        )
        sumWV += Wm * Vm / sqrt(Msub)
        # print(sumWV)
        m += 1
    Asub = Ae + 10 * log(abs(sumWV ** 2), 10)
    # print(Asub)
    # Array excitation
    # Wmn = exp(2j * pi * ((m - 1) * dv * sin(UE_v) - (n - 1) * dh * cos(UE_v) * sin(UE_h))) / sqrt(M * N)
    # Vmn = exp(2j * pi*((m - 1) * dv * cos(S_v) + (n - 1) * dh * sin(S_v) * sin(UE_h)))

    sumWV2 = 0
    m = 1
    # i = 0
    while m <= M:
        n = 1
        while n <= N:
            Wmn = complex(
                cos(2 * pi * ((m - 1) * dv * sin(UE_v) - (n - 1) * dh * cos(UE_v) * sin(UE_h))),
                sin(2 * pi * ((m - 1) * dv * sin(UE_v) - (n - 1) * dh * cos(UE_v) * sin(UE_h)))
            )
            Vmn = complex(
                cos(2 * pi * ((m - 1) * dv * cos(S_v) + (n - 1) * dh * sin(S_v) * sin(S_h))),
                sin(2 * pi * ((m - 1) * dv * cos(S_v) + (n - 1) * dh * sin(S_v) * sin(S_h)))
            )
            sumWV2 += Wmn * Vmn / sqrt(M * N)
            n += 1
            # i+=1
            # print("进来了",i)
        m += 1
    # Composite array radiation pattern
    A_A = Asub + 10 * log(abs(sumWV2 ** 2), 10)
    # print(A_A)
    return A_A


def Ghr(x_h, G180):
    ka = 0.7
    kh = 0.7
    Ykh = 3 * (1 - pow(0.5, -1 * kh))
    G_hr = -12 * x_h * x_h
    if x_h > 0.5:
        G_hr = -12 * pow(x_h, 2 - kh) - Ykh
    if G_hr < G180:
        G_hr = G180
    return G_hr


def Gvr(x_v, v_3dB, G180):
    ka = 0.7
    kv = 0.3
    xk = sqrt(1.33 - 0.33 * kv)
    C = 10 * log10((pow(180 / v_3dB, 1.5) * (pow(4, -1.5) + kv)) / (1 + 8 * ka)) / log10(22.5 / v_3dB)
    Ykv = 12 - C * log10(4) - 10 * log10(pow(4, -1.5) + kv)
    G_vr = 0.0
    if x_v < xk:
        G_vr = -12 * x_v * x_v
    elif x_v < 4:
        G_vr = -15 + 10 * log10(pow(x_v, -1.5) + kv)
    elif x_v < 90 / v_3dB:
        G_vr = -Ykv - 3 - C * log10(x_v)
    elif x_v == 180 / v_3dB:
        G_vr = -G180
    return G_vr


def Gain_TxF1336(S_h, S_v):
    # non-AAS参数:
    # 方位角h，仰角v
    # 角度需要经过下倾角变换后，再代入公式中进行计算
    # 方位角: 范围 -180~180
    # 仰角: 范围 -90~90，变换后与xoy面中x轴的夹角
    ka = 0.7
    h_3dB = 65
    G0 = 16  # 最大天线增益
    v_3dB = 31000 * pow(10, -0.1 * G0) / h_3dB
    x_h = abs(S_h) / h_3dB
    x_v = abs(S_v) / v_3dB
    G180 = -15 + 10 * log10(1 + 8 * ka) - 15 * log10(180 / v_3dB)
    R = (Ghr(x_h, G180) - Ghr(180 / h_3dB, G180)) / (Ghr(0, G180) - Ghr(180 / h_3dB, G180))
    G_hr = Ghr(x_h, G180)
    G_vr = Gvr(x_v, v_3dB, G180)
    return G0 + G_hr + R * G_vr


# 不同类型的天线模型下天线增益的计算
def S456(x):
    # 计算接受增益A_A_get
    if x >= 1 and x < 48:
        A_A_get = 32 - 25 * log10(x)
    elif x >= 48 and x <= 180:
        A_A_get = -10
    return A_A_get


def S580(x):
    if x >= 20 and x < 26.3:
        A_A_get = -3.5
    else:
        A_A_get = 29 - 25 * log10(x)
    return A_A_get


# 计算功率，PFD:power flux density  dB(W/m*m MHz)
def pfd_calc(A_A):
    # pfd = -10 * log10(pow(3 * pow(10, 8) / (4 * pi * 2.6 * 1000), 2)) - A_A + 3 + 60 - 174 + 5 - 6
    pfd = - 82.25085949475593 - A_A
    return pfd


def vec(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return np.array([math.cos(el)*math.cos(az),
                     math.cos(el)*math.sin(az),
                     math.sin(el)])

def az_el_from_vec(v):
    az = math.degrees(math.atan2(v[1], v[0]))
    el = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
    return az, el

def relative_az_el(az_a, el_a, el_b=3,az_b=0):
    vA = vec(az_a, el_a)
    vB = vec(az_b, el_b)

    z = vB
    y = np.array([0, 1, 0])
    x = np.cross(y, z)
    y = np.cross(z, x)          # 重新正交化
    R = np.column_stack([x, y, z])   # 旋转矩阵
    vA_rel = R.T @ vA               # 把 A 旋转到 B 的坐标系
    return az_el_from_vec(vA_rel)

# # 示例：A=(45°, 10°) 相对于 B=(0°, 3°)
# print(relative_az_el(0, -0.0709969078420048))
# 输出 (45.0, 6.96)  表示方位不变，仰角被“拉平”了 3°


# Transformation from an LCS to a GCS for downtilt angle only
def transformV(v, h, tilt):
    # h Horizontal 方位角
    # v vertical仰角
    # tilt 天线的下倾角
    tilt = tilt / 180 * pi
    N_v = acos(cos(h) * sin(v) * sin(tilt) + cos(v) * cos(tilt))
    return N_v


def transformH(v, h, tilt):
    # h Horizontal 方位角
    # v vertical仰角
    # tilt 天线的下倾角
    tilt = tilt / 180 * pi
    N_h = atan2(sin(v) * sin(h),(cos(h) * sin(v) * cos(tilt) - cos(v) * sin(tilt)))
    return N_h


# 针对以地面为原点的极坐标系中的UE(x,h),
# x 极径 0-5000m
# h 方位角 极角 -pi/3~pi/3
# 范围
# 求IMT基站指向UE方向的仰角v和方位角h
def coordinate_transforming_UE(x):
    v = pi - atan(x / (BS_height - BS_ue_height))
    return v


# 根据方位角随机取IMT_UE的极径上限
def IMTUEInCell(_UE_h, BS_radius):
    # 1.732=sqrt(3)
    xlimit = 4000.00
    if _UE_h >= -pi / 3 and _UE_h < -pi / 6:
        xlimit = -1.732 / sin(_UE_h)
    elif _UE_h < 0:
        xlimit = 4 * 1.732 / (-sin(_UE_h) + 1.732 * cos(_UE_h))
    elif _UE_h < pi / 6:
        xlimit = 4 * 1.732 / (sin(_UE_h) + 1.732 * cos(_UE_h))
    elif _UE_h < pi / 3:
        xlimit = 1.732 / sin(_UE_h)
    return xlimit * BS_radius / 4


# 根据两点的经纬度和距地面高度（单位为：m）来计算目标到观察者的方位角和仰角
def calculate_angles(observer_lon, observer_lat, observer_alt,
                     target_lon, target_lat, target_alt):
    # observer_lon,observer_lat,观察者经纬度
    # target_lat, target_lon,目标经纬度
    # observer_alt，target_alt,距地球表面高度,单位为米
    R = 6371.137  # 地球平均半径，单位为公里

    # 将十进制经纬度转化为弧度
    lon1, lat1, lon2, lat2 = map(radians,
                                 [float(observer_lon), float(observer_lat), float(target_lon), float(target_lat)])

    # Convert altitude from meters to kilometers
    alt1 = observer_alt / 1000.0
    alt2 = target_alt / 1000.0

    # Calculate the Cartesian coordinates for both points
    x1 = (R + alt1) * cos(lat1) * cos(lon1)
    y1 = (R + alt1) * cos(lat1) * sin(lon1)
    z1 = (R + alt1) * sin(lat1)

    x2 = (R + alt2) * cos(lat2) * cos(lon2)
    y2 = (R + alt2) * cos(lat2) * sin(lon2)
    z2 = (R + alt2) * sin(lat2)

    # Calculate the vector from observer to target
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # 计算两点之间的水平距离
    horizontal_distance = sqrt(dx ** 2 + dy ** 2)

    # 计算方位角（方位）角
    azimuth = atan2(dy, dx)

    # 计算仰角
    elevation = atan2(dz, horizontal_distance)

    # 把角度从弧度换算成度数
    azimuth_deg = azimuth * pi / 180
    elevation_deg = elevation * pi / 180

    return azimuth_deg, elevation_deg




def calculate_angle(vector1, vector2):
    """
    计算两个向量的夹角
    :param vector1: 第一个向量 (numpy数组)
    :param vector2: 第二个向量 (numpy数组)
    :return: 夹角 (单位：度)
    """
    # 确保输入是numpy数组
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # 计算向量的点积
    dot_product = np.dot(vector1, vector2)

    # 计算向量的模
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)

    # 计算夹角的余弦值
    cos_angle = dot_product / (magnitude_vector1 * magnitude_vector2)

    # 计算夹角
    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)  # 转换为度

    return angle


# # 示例向量
# vector1 = [1, 2, 3]
# vector2 = [4, 5, 6]
#
# # 计算夹角
# angle = calculate_angle(vector1, vector2)
# print(f"夹角: {angle:.2f} 度")


# R=32.4+20×log(D)+20×log(M)
# 其中，D是无线信号自由空间传播距离，单位为km；
# M是频率MHz；
# R是信号损耗值，单位为dBm。
def path_loss(DKM, M):
    """
    计算路径损耗。
    参数:
    DKM -- 无线信号自由空间传播距离，单位为千米
    M -- 频率，单位为兆赫兹
    返回:
    R -- 信号损耗值，单位为dBm
    """
    # 计算路径损耗
    path_loss = 32.4 + 20 * log10(DKM) + 20 * log10(M)

    return path_loss

def clutter_loss(fGHz: float, p: float, d: float):
    """
        计算宏基站杂波损耗（clutter loss）。

        参数
        ----
        f : float
            频率 (GHz)
        p : float
            概率值 (0 < p < 1)
        d : float
            距离 (km)

        返回
        ----
        Lclt : float
            杂波损耗 (dB)
        """

    sigmas = 6.0
    sigmal = 4.0

    Ll = -2 * math.log10(10 ** (-5 * math.log10(fGHz) - 12.5) + 10 ** (-16.5))
    Ls = 32.98 + 23.9 * math.log10(d) + 3 * math.log10(fGHz)

    w = 10 ** (-0.2 * Ll)
    v = 10 ** (-0.2 * Ls)
    sigmacb = math.sqrt((sigmal ** 2 * w + sigmas ** 2 * v) / (w + v))

    Lclt = -5 * math.log10(w + v) - sigmacb * norm.ppf(1-p)

    return Lclt

# ITU-R F.699 仅适用于 D/λ <= 100 的大孔径天线！
def itu_r_f699_gain(theta_deg, D_m, freq_Hz,Gmax):
    """
    ITU-R F.699-7 固定业务小孔径天线方向图（D/λ <= 100）
    Parameters
    ----------
    theta_deg : float or np.ndarray
        离轴角 [deg]
    D_m : float
        天线直径 [m]
    freq_Hz : float
        工作频率 [Hz]

    Returns
    -------
    G_dBi : float or np.ndarray
        天线增益 [dBi]
    """
    c = 299792458.0          # 光速 [m/s]
    lam = c / freq_Hz        # 波长 [m]
    D_lam = D_m / lam        # D/λ
    G1=2+15*log10(D_lam)
    # print(G1)
    theta_deg_m=20/D_lam*sqrt(Gmax-G1)

    # 最大增益（典型抛物面，η=0.7）
    eta = 0.7
    G_max = 10 * np.log10(eta * (np.pi * D_lam) ** 2)

    # 模型关键角度
    theta = np.atleast_1d(theta_deg)
    theta_rad = np.radians(theta)

    # 分区计算
    G = np.full_like(theta, 10-10*log10(D_lam), dtype=float)  # 先填远区 -10 dBi

    # 主瓣区
    mask_main = theta <= theta_deg_m
    G[mask_main] = G_max - 2.5e-3 * (D_lam * theta[mask_main]) ** 2

    # 过渡区
    theta_r_deg = 100/D_lam
    mask_trans = (theta > theta_deg_m) & (theta < theta_r_deg)
    G[mask_trans] = G1

    # 旁瓣区
    mask_side = (theta >= theta_r_deg) & (theta <= 48.0)
    G[mask_side] = 52 -10*log10(D_lam)-25 * np.log10(theta[mask_side])

    # 远区已预设为 10-10*log10(D_lam)

    # 若输入为标量，返回标量
    if np.isscalar(theta_deg):
        return G.item()
    return G

#
# # ---------------- 使用示例 ----------------
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     D = 60          # 60 m
#     f = 2010         # 频率Hz
#     theta = np.linspace(0, 180, 1000)
#     gain = itu_r_f699_gain(theta, D, f,19.4)
#
#     plt.plot(theta, gain)
#     plt.xlabel("Off-axis angle θ [deg]")
#     plt.ylabel("Gain [dBi]")
#     plt.title("ITU-R F.699-7 Antenna Pattern\nD = 1.2 m, f = 14 GHz")
#     plt.grid(True)
#     plt.show()


