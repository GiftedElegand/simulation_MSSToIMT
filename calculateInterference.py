from math import pi, sqrt, log10, cos, sin
import random
import matplotlib.pyplot as plt
from dask.array import arctan
from IMT_Simulation import IMTUEInCell, coordinate_transforming_UE, transformV, transformH, gain_calc, path_loss, \
    calculate_angle, S580, Gain_TxF1336, clutter_loss, itu_r_f699_gain, relative_az_el, clutter_loss_with_max_limit
from Parm_Simulation import BS_radius, Robservertotarget, BS_tilt, BS_height, isdis, BS_ue_height, band, \
    ACLR, station_height, Body_loss, Satellite_dis

import numpy as np

# 创建极坐标轴
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

import math

from calculate import Cal_Gain_tx_M2101


def calculate_azimuth_elevation(x1, y1, z1, x2, y2, z2):
    """
    计算 A -> B 的方位角 azimuth 和仰角 elevation
    方位角范围：-180° ~ +180°（正为 0°，顺时针为正）
    参数
    ----
    A, B : 长度为 3 的可迭代对象 (x, y, z)
    返回角度制
     ----
    (azimuth, elevation) : 设定返回角度
    """

    # 计算两点之间的向量
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # 水平距离
    d_hor = math.hypot(dx, dy)

    # 计算从东顺时针的方位角（弧度）
    azimuth = math.atan2(dy,dx)  # 返回弧度值
    # azimuth = math.degrees(azimuth)  # 转换为度

    # 计算仰角
    elevation = math.atan2(dz, d_hor)
    # elevation = math.degrees(elevation)  # 转换为度

    return azimuth, elevation


# IMT基站和干扰方关系：挖洞
def singleInterference1():
    # 确定IMT UE的位置
    _UE_h = random.uniform(-60, 60) / 180 * pi
    UE_x = random.uniform(0, IMTUEInCell(_UE_h, BS_radius))
    _UE_v = coordinate_transforming_UE(UE_x)
    print(_UE_h, _UE_v)
    # 确定卫星 UE的位置
    s_UE_h = random.uniform(-180, 180) / 180 * pi
    s_x = random.uniform(isdis, Robservertotarget / 2)
    s_UE_v = coordinate_transforming_UE(s_x)

    # 天线+下倾角之后的坐标转化
    UE_v = transformV(_UE_v, _UE_h, BS_tilt) - pi / 2
    UE_h = transformH(_UE_v, _UE_h, BS_tilt)
    S_v = transformV(s_UE_v, s_UE_h, BS_tilt)
    S_h = transformH(s_UE_v, s_UE_h, BS_tilt)
    # 计算发射增益A_A_out=-3
    # 计算接受增益A_A_get
    A_A_get = gain_calc(UE_v, UE_h, S_v, S_h)
    # 计算路径损耗path_loss
    # 计算集总干扰Interference
    interference = 23 - 3 + A_A_get - path_loss(sqrt(s_x ** 2 + (BS_height - BS_ue_height) ** 2) / 1000, band)
    plt.text(s_UE_h, s_x + 0.2, round(interference, 2), ha='center', va='bottom', fontsize=8)
    return interference


# 特殊情况计算干扰方UE在IMT基站正上方，IMT的UE在天线面板的法线方向
def singA_A_getS1(tit):
    # 确定IMT UE的位置
    _UE_h = 0
    _UE_v = (90 + tit) / 180 * pi
    print(_UE_h, _UE_v * 180 / pi)
    # ax.plot(_UE_h, UE_x, color="blue", marker="o")
    # 确定干扰方UE的位置
    s_UE_h = 0
    s_UE_v = 0
    print(s_UE_h, s_UE_v)
    # 天线+下倾角之后的坐标转化
    UE_v = transformV(_UE_v, _UE_h, tit)
    UE_h = transformH(_UE_v, _UE_h, tit)
    S_v = transformV(s_UE_v, s_UE_h, tit)
    S_h = transformH(s_UE_v, s_UE_h, tit)
    # 计算接受增益A_A_get
    A_A_get = gain_calc(UE_v, UE_h, S_v, S_h)
    print(
        f"IMTUE = {UE_v / pi * 180:.2f}，{UE_h:.2f}, 干扰UE = {S_v / pi * 180:.2f}，{S_h:.2f}, 接受增益 = {A_A_get:.2f}")
    return A_A_get


# 受干扰基站和干扰方关系：拉远（用于多次循环，无输出图；接收增益为2101模型和发射天线为全向模型）可用于HIBS系统
def singleInterference3():
    # 确定受干扰方 UE的位置
    _UE_h = random.uniform(-60, 60) / 180 * pi
    UE_x = random.uniform(0, 100000)
    _UE_v = coordinate_transforming_UE(UE_x)
    IMT_station = (0, 0)
    # IMT_UE = (UE_x*cos(_UE_h), UE_x*sin(_UE_h))
    # 确定卫星 UE的位置
    s_UE_h = random.uniform(-180, 180) / 180 * pi
    s_x = random.uniform(0, Robservertotarget / 2)
    Satellite_UE = (Satellite_dis + s_x * cos(s_UE_h), s_x * sin(s_UE_h))
    # 找卫星UE和被干扰站之间的关系
    # IMT指向卫星UE在水平面的投影距离
    IMT_to_SUE_x = sqrt(Satellite_UE[0] ** 2 + Satellite_UE[1] ** 2)
    if IMT_to_SUE_x >= Robservertotarget:
        return -1000

    # IMT指向卫星UE的方位角h，仰角v
    IMT_to_SUE_h, IMT_to_SUE_v = calculate_azimuth_elevation(Satellite_UE[0], Satellite_UE[1], 1.5, IMT_station[0],
                                                             IMT_station[1], BS_height)
    IMT_to_SUE_h = arctan(Satellite_UE[1] / Satellite_UE[0])
    # _UE_h, _UE_v = calculate_azimuth_elevation(0, 0, 0, 1, 0, 0)
    # IMT_to_SUE_v=coordinate_transforming_UE(IMT_to_SUE_x)
    # print(math.degrees(_UE_v), math.degrees(_UE_h), math.degrees(IMT_to_SUE_v), math.degrees(IMT_to_SUE_h))
    # 天线+下倾角之后的坐标转化
    UE_v = transformV(_UE_v, _UE_h, BS_tilt) - pi / 2
    UE_h = transformH(_UE_v, _UE_h, BS_tilt)
    S_v = transformV(IMT_to_SUE_v, IMT_to_SUE_h, BS_tilt)
    S_h = transformH(IMT_to_SUE_v, IMT_to_SUE_h, BS_tilt)
    # print(math.degrees(UE_v), math.degrees(UE_h), math.degrees(S_v), math.degrees(S_h))
    # MSS UE天线指向水平的指向撒点的原点中心
    # Satellite_vector = [Satellite_dis - Satellite_UE[0], -Satellite_UE[1], 0]
    # Satellite_vector = [Satellite_dis - Satellite_UE[0], -Satellite_UE[1], 0]
    # temp1 = random.uniform(-180, 180) / 180 * pi
    # temp2 = 5 / 180 * pi
    # Satellite_vector = [cos(temp2) * cos(temp1), cos(temp2) * sin(temp1), sin(temp2)]
    # MSS UE天线指向受扰方基站天线的向量坐标
    # stationary = [-Satellite_UE[0], -Satellite_UE[1], BS_height - BS_ue_height]

    # 两个向量的夹角x
    # x = calculate_angle(Satellite_vector, stationary)

    # 计算发射增益A_A_out
    # A_A_out = S580(x) - 23
    A_A_out = -5

    # 计算接受增益A_A_get
    # A_A_get = gain_calc(UE_v, UE_h, S_v, S_h)
    A_A_get = gain_calc(UE_v, UE_h, S_v, S_h)
    # print(math.degrees(UE_v), UE_h, S_v, S_h)
    # 计算路径损耗path_loss
    d = sqrt(IMT_to_SUE_x ** 2 + (BS_height - BS_ue_height) ** 2) / 1000

    Ploss = path_loss(d, band)
    # closs = clutter_loss(band / 1000, random.uniform(0, 1), d)
    # clossmax = clutter_loss(band / 1000, random.uniform(0, 1), 2)
    # if clossmax < closs:
    #     closs = clossmax

    # 计算单星干扰Interference
    interference = 33.4 + A_A_get + A_A_out - Ploss - ACLR - Body_loss+10

    print(f"接收增益 = {A_A_get:.2f}, 路径损耗 = {Ploss:.2f}, 干扰 = {interference:.2f}")
    return interference


# 卫星终端对其他系统的干扰：射电天文，卫星气象
def singleInterferenceOther():
    # 受干扰方IMT UE的位置
    x = 0
    # 确定卫星 UE的位置
    s_UE_h = random.uniform(-180, 180) / 180 * pi
    s_x = random.uniform(0, Robservertotarget / 2)
    # MSS UE天线的向量坐标为
    Satellite_vector = [Satellite_dis + s_x * cos(s_UE_h), s_x * sin(s_UE_h), BS_ue_height]
    # MSS UE天线指向受扰方基站天线的向量坐标（-x, 0, h1 - h2）
    stationary = [-Satellite_dis, 0, BS_height - BS_ue_height]
    # 找卫星UE和被干扰站之间的关系
    # IMT指向卫星UE在水平面的投影距离
    IMT_to_SUE_x = sqrt(Satellite_vector[0] ** 2 + Satellite_vector[1] ** 2)
    if IMT_to_SUE_x >= Robservertotarget:
        return -1000

    # 两个向量的夹角x
    x = calculate_angle(Satellite_vector, stationary)
    # IMT_to_SUE_v=coordinate_transforming_UE(IMT_to_SUE_x)
    # 计算发射增益A_A_out=-3
    A_A_out = S580(x)
    # 计算接受增益A_A_get

    Ploss = path_loss(sqrt(IMT_to_SUE_x ** 2 + (BS_height - BS_ue_height) ** 2) / 1000, band)
    # 计算路径损耗path_loss
    # 计算单星干扰Interference
    interference = 33.4 + 6 + A_A_out - Ploss - ACLR - Body_loss
    # ax.plot(_UE_h, UE_x, color="blue", marker="o")
    # ax.plot(IMT_to_SUE_h, IMT_to_SUE_x, color="red", marker="^")
    # plt.text(IMT_to_SUE_h, IMT_to_SUE_x + 0.2, round(interference, 2), ha='center', va='bottom', fontsize=8)
    # print(f"接收增益 = {A_A_get:.2f}, 路径损耗 = {Ploss:.2f}, 干扰 = {interference:.2f}")
    return interference


# IMT基站和干扰方关系：拉远（用于 多次循环，无输出图）
def singleInterference2():
    # 确定IMT UE的位置
    _UE_h = random.uniform(-60, 60) / 180 * pi
    UE_x = random.uniform(0, IMTUEInCell(_UE_h, BS_radius))
    _UE_v = coordinate_transforming_UE(UE_x)
    IMT_station = (0, 0)
    IMT_UE = (UE_x * cos(_UE_h), UE_x * sin(_UE_h))
    # 确定卫星 UE的位置
    s_UE_h = random.uniform(-180, 180) / 180 * pi
    s_x = random.uniform(0, Robservertotarget / 2)
    Satellite_UE = (Satellite_dis + s_x * cos(s_UE_h), s_x * sin(s_UE_h))
    # 找卫星UE和被干扰站之间的关系
    # IMT指向卫星UE在水平面的投影距离
    IMT_to_SUE_x = sqrt(Satellite_UE[0] ** 2 + Satellite_UE[1] ** 2)
    if IMT_to_SUE_x >= Robservertotarget:
        return -1000

    # IMT指向卫星UE的方位角h，仰角v
    IMT_to_SUE_h, IMT_to_SUE_v = calculate_azimuth_elevation(Satellite_UE[0], Satellite_UE[1], 1.5, IMT_station[0],
                                                             IMT_station[1], BS_height)
    IMT_to_SUE_h = arctan(Satellite_UE[1] / Satellite_UE[0])
    # IMT_to_SUE_v=coordinate_transforming_UE(IMT_to_SUE_x)
    # 天线+下倾角之后的坐标转化
    UE_v = transformV(_UE_v, _UE_h, BS_tilt) - pi / 2
    UE_h = transformH(_UE_v, _UE_h, BS_tilt)
    S_v = transformV(IMT_to_SUE_v, IMT_to_SUE_h, BS_tilt)
    S_h = transformH(IMT_to_SUE_v, IMT_to_SUE_h, BS_tilt)
    # 计算发射增益A_A_out=-3
    # 计算接受增益A_A_get
    A_A_get = gain_calc(UE_v, UE_h, S_v, S_h)
    Ploss = path_loss(sqrt(IMT_to_SUE_x ** 2 + (BS_height - BS_ue_height) ** 2) / 1000, band)
    # 计算路径损耗path_loss
    # 计算单星干扰Interference
    interference = 45 - 3 + A_A_get - Ploss - ACLR - Body_loss
    return interference


# IMT终端和干扰方关系：拉远（用于多次循环，无输出图;接收天线为全向天线）
def singleInterference4():
    # 确定IMT UE的位置
    _UE_h = random.uniform(-60, 60) / 180 * pi
    UE_x = random.uniform(0, IMTUEInCell(_UE_h, BS_radius))
    _UE_v = coordinate_transforming_UE(UE_x)
    IMT_station = (0, 0)
    IMT_UE = (UE_x * cos(_UE_h), UE_x * sin(_UE_h))
    # 确定卫星 UE的位置
    s_UE_h = random.uniform(-180, 180) / 180 * pi
    s_x = random.uniform(0, Robservertotarget / 2)
    Satellite_UE = (Satellite_dis + s_x * cos(s_UE_h), s_x * sin(s_UE_h))
    # 找卫星UE和被干扰站之间的关系
    # IMT UE指向卫星UE在水平面的投影距离
    IMT_to_SUE_x = sqrt(Satellite_UE[0] ** 2 + Satellite_UE[1] ** 2)
    if IMT_to_SUE_x >= Robservertotarget:
        return -1000
    Ploss = path_loss(IMT_to_SUE_x / 1000, band)
    # 计算路径损耗path_loss
    # 计算单星干扰Interference
    interference = 33.4 - 5 - 3 - Ploss - ACLR - Body_loss
    return interference


# IMT基站和干扰方关系：拉远（用于多次循环，无输出图；接收增益为1336模型和发射天线为全向模型）
def singleInterference6():
    IMT_station = (0, 0)
    # 确定卫星 UE的位置
    s_UE_h = random.uniform(-pi/2, pi/2)
    # s_UE_h=0
    s_x = random.uniform(0, Robservertotarget / 2)
    # s_x=0
    # s_x = Satellite_dis
    Satellite_UE = (s_x * cos(s_UE_h)+Satellite_dis, s_x * sin(s_UE_h))
    # print(Satellite_UE)
    # 找卫星UE和被干扰站之间的关系
    # IMT指向卫星UE在水平面的投影距离
    IMT_to_SUE_x = sqrt(Satellite_UE[0] ** 2 + Satellite_UE[1] ** 2)
    # IMT_to_SUE_x = Satellite_dis
    # if IMT_to_SUE_x > Robservertotarget:
    #     return -1000
    # IMT指向卫星UE的方位角h，仰角v
    IMT_to_SUE_h, IMT_to_SUE_v = calculate_azimuth_elevation(IMT_station[0],IMT_station[1], BS_height, Satellite_UE[0], Satellite_UE[1], 1.5)
    # print(math.degrees(IMT_to_SUE_h), math.degrees(IMT_to_SUE_v))
    # 天线+下倾角之后的坐标转化
    # S_h = transformH(IMT_to_SUE_v, IMT_to_SUE_h, BS_tilt)
    # S_v = transformV(IMT_to_SUE_v, IMT_to_SUE_h, BS_tilt)
    S_h,S_v=relative_az_el(math.degrees(IMT_to_SUE_h), math.degrees(IMT_to_SUE_v),90-BS_tilt)
    # print(math.degrees(S_h),math.degrees(S_v))
    # print(S_h,S_v)
    # MSS UE天线指向原点
    # Satellite_vector = [Satellite_dis + s_x * cos(s_UE_h), s_x * sin(s_UE_h), BS_ue_height]
    # MSS UE天线指向水平的指向撒点的原点中心
    # Satellite_vector = [Satellite_dis - Satellite_UE[0], -Satellite_UE[1], 0]
    # Satellite_vector = [Satellite_dis - Satellite_UE[0], -Satellite_UE[1], 0]
    # temp1 = random.uniform(-180, 180) / 180 * pi
    # temp2 = 5/ 180 * pi
    # Satellite_vector = [cos(temp2)*cos(temp1),cos(temp2)*sin(temp1),sin(temp2)]
    # MSS UE天线指向受扰方基站天线的向量坐标
    stationary = [-Satellite_UE[0],-Satellite_UE[1], BS_height-BS_ue_height]
    # FS的天线指向水平的指向随意方向
    # temp=random.uniform(0, 2*pi)
    # FS_vector=[cos(temp), sin(temp), 0]

    # 发射增益两个向量的夹角x
    # x = calculate_angle(Satellite_vector, stationary)


    # 计算发射增益A_A_out
    # A_A_out = S580(x) - 23
    A_A_out = 0

    # 计算接受增益A_A_get
    A_A_get = Gain_TxF1336(S_h, S_v)
    # 发射增益两个向量的夹角x
    # x1 = calculate_angle(FS_vector, stationary)
    # A_A_get = itu_r_f699_gain(x1,60/100, 2010e6, 19.4)


    # 计算路径损耗path_loss
    d = sqrt(IMT_to_SUE_x ** 2 + (BS_height - BS_ue_height) ** 2) / 1000
    # d=IMT_to_SUE_x/1000
    Ploss = path_loss(d, band)

    closs = clutter_loss(band / 1000, random.uniform(0, 1), d)
    clossmax = clutter_loss(band / 1000, random.uniform(0, 1), 2)
    if clossmax < closs:
        closs = clossmax
    # 计算单星干扰Interference
    interference = 33.4 + A_A_get + A_A_out - Ploss  - Body_loss - closs+4
    print(f"卫星UE的位置={Satellite_UE},接收增益 = {A_A_get:.2f}, 路径损耗 = {Ploss:.2f}, 干扰 = {interference:.2f}")
    return interference

# IMT基站和干扰方关系：拉远（用于多次循环，无输出图；接收增益为1336模型和发射天线为全向模型）
def singleUEToUEInterference():
    # 确定卫星 UE的位置
    s_UE_h = random.uniform(-pi/2, pi/2)
    s_x = random.uniform(0, Robservertotarget / 2)
    # 卫星 UE 极坐标转换成直角坐标
    Satellite_UE = (s_x * cos(s_UE_h), s_x * sin(s_UE_h))
    # print(Satellite_UE)
    # 找卫星UE和被干扰站之间的关系:固定卫星星下点在原点，移动受扰站位置
    #IMT UE的位置
    IMT_UE=(Satellite_dis,0)
    # IMT指向卫星UE在水平面的投影距离
    IMT_to_SUE_x = sqrt((IMT_UE[0]-Satellite_UE[0]) ** 2 + Satellite_UE[1] ** 2)
    # 计算发射增益A_A_out
    A_A_out = 0
    # 计算接受增益A_A_get
    A_A_get = -3
    # 计算路径损耗path_loss
    d = sqrt(IMT_to_SUE_x ** 2 + (BS_height - BS_ue_height) ** 2) / 1000
    # d=IMT_to_SUE_x/1000
    Ploss = path_loss(d, band)
    closs = clutter_loss_with_max_limit(band / 1000, random.uniform(0, 1), d)
    # 计算单星干扰Interference
    interference = 33.4 + A_A_get + A_A_out - Ploss  - Body_loss - closs
    # print(f"卫星UE的位置={Satellite_UE},接收增益 = {A_A_get:.2f}, 路径损耗 = {Ploss:.2f}, 干扰 = {interference:.2f}")
    return interference


# 求集总干扰
def sumInterference(num):
    sum = 0
    for i in range(num):
        sum += 10 ** (singleUEToUEInterference() * 0.1)
    answer = 10 * log10(sum)
    # print(answer)
    return answer