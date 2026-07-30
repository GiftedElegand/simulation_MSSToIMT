# 单位，m
from math import acos, sqrt, pi, floor, ceil


def max_terminals_hex(R: float, r: float) -> int:
    """
    计算在半径为 R 的圆内，以六边形蜂窝方式部署直径为 r 的小区时，
    可放置的终端数量的下界（假设圆心与一个蜂窝中心重合，
    且只考虑完整的六边形环）。
    参数:
        R : float  部署范围圆半径
        r : float  小区直径
    返回:
        int  最大终端数量
    """
    if R <= 0 or r <= 0:
        return 0
    cell_radius = r / 2.0
    # 六边形蜂窝层间距
    layer_spacing = sqrt(3) * cell_radius
    max_layers = int(R / layer_spacing)
    # 总终端数 = 1 + 3 * n * (n + 1)
    return 1 + 3 * max_layers * (max_layers + 1)

band = 1900  # 频段
Satellite_height = 670  # Satellite站高，单位km
# 同频ACLR=0
# 邻频IMT基站ACLR=45 IMT终端ACLR=30
ACLR=0
# IMT参数(农村场景）
BS_radius = 4000  # IMT小区直径m
BS_height = 20000  # IMT基站高度m
station_height=55 # 其他系统站高m
BS_ue_height = 1.5  # IMT终端高度m
BS_tilt = 3 # IMT天线下倾角
# 隔离（挖孔）距离
isdis = 0
# 卫星终端密度
SatelliteUEdensity=1/660000000
Body_loss=4 #单位db

R = 6371137  # 地球平均半径，单位为m
# 对于IMT基站卫星的可见半径 单位为m 
Robservertotarget = acos(R / (R + BS_ue_height)) * R + acos(R / (R + BS_ue_height)) * R
print(Robservertotarget)
# 隔离距离
SeparationDistance=10000
Satellite_dis =SeparationDistance+Robservertotarget/2  # Satellite星下点距IMT基站距离，单位m
# print(Satellite_dis)
# Satellite_dis = 23000
# 计算卫星终端的个数
# 可见面积
VisibleArea=pi*(Robservertotarget - isdis)**2
# range_SUE_num = floor(VisibleArea * SatelliteUEdensity)  # 向下取整得到整数个数
range_SUE_num=ceil(VisibleArea * SatelliteUEdensity) # 向上取整得到整数个数
# range_SUE_num = max_terminals_hex(Robservertotarget,BS_radius)
# range_SUE_num=1
# print(VisibleArea)
print(range_SUE_num)

