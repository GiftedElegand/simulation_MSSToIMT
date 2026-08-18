"""
卫星通信与地面 IMT 系统干扰仿真计算模块（新版本 - 参数化配置）

本模块提供基于场景配置的干扰计算功能，支持干扰站和受扰站参数的灵活配置。
所有参数集中管理，便于仿真计算和参数调整。

核心公式：Interference = P_tx + G_tx + G_rx - PL - ACLR - Body_loss
"""

from math import pi, sqrt, log10, cos, sin, atan2, degrees, radians
import random
import matplotlib.pyplot as plt
import numpy as np

# 从原模块导入必要的函数和参数
from IMT_Simulation import (
    IMTUEInCell, coordinate_transforming_UE, transformV, transformH,
    gain_calc, path_loss, calculate_angle, S580, Gain_TxF1336,
    clutter_loss, itu_r_f699_gain, relative_az_el
)
from Parm_Simulation import (
    BS_radius, Robservertotarget, BS_tilt, BS_height, isdis,
    BS_ue_height, band, ACLR, station_height, Body_loss, Satellite_dis
)
from calculate import Cal_Gain_tx_M2101


# ============================================================================
# 干扰站（卫星终端）参数配置字典
# 格式：场景名：{发射功率 (dBm), 发射增益 (dBi), 描述}
# ============================================================================
INTERFERENCE_STATION_CONFIGS = {
    "scenario1": {
        "tx_power": 23,      # 发射功率 (dBm)
        "tx_gain": -3,       # 发射增益 (dBi)
        "description": "IMT 基站挖洞场景 - 保护区域计算"
    },
    "scenario2": {
        "tx_power": 45,      # 发射功率 (dBm)
        "tx_gain": -3,       # 发射增益 (dBi)
        "description": "拉远场景 - 高功率终端累积干扰评估"
    },
    "scenario4": {
        "tx_power": 33.4,    # 发射功率 (dBm)
        "tx_gain": -5,       # 发射增益 (dBi)
        "description": "卫星对 IMT 用户终端干扰"
    },
    "scenario6": {
        "tx_power": 30,      # 发射功率 (dBm)
        "tx_gain": -5,       # 发射增益 (dBi)
        "description": "固定业务/射电天文保护场景"
    },
    "scenario_other": {
        "tx_power": 33.4,    # 发射功率 (dBm)
        "tx_gain": 6,        # 发射增益 (dBi) - S580 模型计算后使用
        "description": "射电天文/卫星气象敏感系统保护"
    }
}


# ============================================================================
# 受扰站参数配置字典
# 格式：场景名：{接收天线增益 (dBi), 天线模型，噪声系数 (dB), 描述}
# 注意：rx_gain 可以是固定值、None(动态计算) 或函数字符串
# ============================================================================
VICTIM_STATION_CONFIGS = {
    "scenario1": {
        "rx_gain": None,              # 动态计算 (通过 gain_calc)
        "antenna_model": "gain_calc", # IMT 基站天线模型 (AAS)
        "noise_figure": 5.0,          # 噪声系数 (dB)
        "description": "IMT 基站 - AAS 天线模型"
    },
    "scenario3": {
        "rx_gain": None,              # 动态计算 (通过 gain_calc)
        "antenna_model": "gain_calc", # IMT/HIBS 基站天线模型
        "noise_figure": 5.0,          # 噪声系数 (dB)
        "description": "HIBS/远距离基站 - AAS 天线模型"
    },
    "scenario4": {
        "rx_gain": -3,                # 固定增益 (全向天线)
        "antenna_model": "omni",      # 全向天线模型
        "noise_figure": 7.0,          # 噪声系数 (dB) - 用户终端通常较高
        "description": "IMT 用户终端 - 全向天线"
    },
    "scenario6": {
        "rx_gain": None,              # 动态计算 (通过 F1336 模型)
        "antenna_model": "F1336",     # ITU-R F.1336 固定业务天线模型
        "noise_figure": 4.0,          # 噪声系数 (dB)
        "description": "固定业务基站 - ITU-R F.1336 天线模型"
    },
    "scenario_other": {
        "rx_gain": None,              # 动态计算 (通过 S580 模型)
        "antenna_model": "S580",      # ITU-R S.580 卫星天线模型
        "noise_figure": 3.0,          # 噪声系数 (dB) - 射电天文要求极低噪声
        "description": "射电天文/卫星气象 - S.580 天线模型"
    }
}


def get_interference_params(scenario_name):
    """
    获取指定场景的干扰站参数

    参数
    ----
    scenario_name : str
        场景名称，如 "scenario1", "scenario2" 等

    返回
    ----
    dict : 包含 tx_power, tx_gain, description 的字典
    """
    if scenario_name not in INTERFERENCE_STATION_CONFIGS:
        raise ValueError(f"未知场景：{scenario_name}. 可用场景：{list(INTERFERENCE_STATION_CONFIGS.keys())}")

    return INTERFERENCE_STATION_CONFIGS[scenario_name].copy()


def get_victim_params(scenario_name):
    """
    获取指定场景的受扰站参数

    参数
    ----
    scenario_name : str
        场景名称，如 "scenario1", "scenario2" 等

    返回
    ----
    dict : 包含 rx_gain, antenna_model, noise_figure, description 的字典
    """
    if scenario_name not in VICTIM_STATION_CONFIGS:
        raise ValueError(f"未知场景：{scenario_name}. 可用场景：{list(VICTIM_STATION_CONFIGS.keys())}")

    return VICTIM_STATION_CONFIGS[scenario_name].copy()


def calculate_azimuth_elevation(x1, y1, z1, x2, y2, z2):
    """
    计算 A -> B 的方位角 azimuth 和仰角 elevation

    参数
    ----
    x1, y1, z1 : float
        起点坐标
    x2, y2, z2 : float
        终点坐标

    返回
    ----
    (azimuth, elevation) : tuple
        方位角和仰角（弧度制）
    """
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    d_hor = sqrt(dx**2 + dy**2)
    azimuth = atan2(dy, dx)
    elevation = atan2(dz, d_hor)

    return azimuth, elevation


def calculate_rx_gain(victim_config, UE_v, UE_h, S_v, S_h, **kwargs):
    """
    根据受扰站配置计算接收增益

    参数
    ----
    victim_config : dict
        受扰站配置字典
    UE_v, UE_h : float
        受扰站天线的垂直和水平角度
    S_v, S_h : float
        干扰源相对于受扰站的角度
    **kwargs : dict
        其他可选参数（如距离、频率等）

    返回
    ----
    float : 接收增益 (dBi)
    """
    antenna_model = victim_config["antenna_model"]
    rx_gain_fixed = victim_config["rx_gain"]

    # 如果配置了固定增益，直接返回
    if rx_gain_fixed is not None:
        return rx_gain_fixed

    # 根据天线模型动态计算
    if antenna_model == "gain_calc":
        return gain_calc(UE_v, UE_h, S_v, S_h)
    elif antenna_model == "F1336":
        # ITU-R F.1336 模型
        az = degrees(S_h)
        el = degrees(S_v)
        return Gain_TxF1336(az, el, **kwargs)
    elif antenna_model == "S580":
        # ITU-R S.580 模型
        angle = kwargs.get("angle", 0)
        return S580(angle)
    elif antenna_model == "omni":
        return -3  # 全向天线默认增益
    else:
        raise ValueError(f"未知的天线模型：{antenna_model}")


def calculate_single_interference(
    int_scenario_name,
    vic_scenario_name,
    tx_power=None,
    tx_gain=None,
    rx_gain_fixed=None,
    antenna_model=None,
    victim_distance_range=None,
    satellite_distance_range=None,
    is_ue_victim=False,
    use_relative_az_el=False,
    extra_gain_offset=0,
    custom_tx_gain_func=None
):
    """
    通用单干扰源计算函数（合并版本）

    通过参数配置支持所有场景的计算，避免代码重复

    参数
    ----
    scenario_name : str
        场景名称，用于获取默认配置
    tx_power : float, optional
        发射功率 (dBm)，如果不传则使用场景默认值
    tx_gain : float, optional
        发射增益 (dBi)，如果不传则使用场景默认值
    rx_gain_fixed : float, optional
        接收增益固定值，如果不传则根据天线模型动态计算
    antenna_model : str, optional
        天线模型，如果不传则使用场景默认值
    victim_distance_range : tuple, optional
        受害方距离范围 (min, max)，单位：米
    satellite_distance_range : tuple, optional
        卫星终端距离范围 (min, max)，单位：米
    is_ue_victim : bool, default=False
        受害方是否为用户终端（True）还是基站（False）
    use_relative_az_el : bool, default=False
        是否使用相对方位角/仰角转换
    extra_gain_offset : float, default=0
        额外的增益偏移量 (dB)
    custom_tx_gain_func : callable, optional
        自定义发射增益计算函数

    返回
    ----
    float : 干扰值 (dBm)
    """
    # 获取默认配置
    int_params = get_interference_params(int_scenario_name)
    vic_params = get_victim_params(vic_scenario_name)

    # 使用传入参数或默认值
    if tx_power is None:
        tx_power = int_params["tx_power"]
    if tx_gain is None:
        tx_gain = int_params["tx_gain"]
    if antenna_model is None:
        antenna_model = vic_params["antenna_model"]

    # 构建临时的受扰站配置
    temp_vic_config = vic_params.copy()
    if rx_gain_fixed is not None:
        temp_vic_config["rx_gain"] = rx_gain_fixed
    if antenna_model != vic_params["antenna_model"]:
        temp_vic_config["antenna_model"] = antenna_model

    # 确定 IMT UE 的位置
    _UE_h = random.uniform(-60, 60) / 180 * pi

    if victim_distance_range:
        UE_x = random.uniform(victim_distance_range[0], victim_distance_range[1])
    elif is_ue_victim:
        UE_x = random.uniform(0, IMTUEInCell(_UE_h, BS_radius))
    else:
        UE_x = random.uniform(0, IMTUEInCell(_UE_h, BS_radius))

    _UE_v = coordinate_transforming_UE(UE_x)

    if is_ue_victim:
        IMT_station = (0, 0)
        IMT_UE = (UE_x * cos(_UE_h), UE_x * sin(_UE_h))

    # 确定卫星 UE 的位置
    s_UE_h = random.uniform(-180, 180) / 180 * pi

    if satellite_distance_range:
        s_x = random.uniform(satellite_distance_range[0], satellite_distance_range[1])
    else:
        s_x = random.uniform(0, Robservertotarget / 2)

    if use_relative_az_el:
        # scenario6 的特殊处理
        Satellite_UE = (s_x * cos(s_UE_h) + Satellite_dis, s_x * sin(s_UE_h))
    else:
        Satellite_UE = (Satellite_dis + s_x * cos(s_UE_h), s_x * sin(s_UE_h))

    # 计算距离
    IMT_to_SUE_x = sqrt(Satellite_UE[0]**2 + Satellite_UE[1]**2)

    if IMT_to_SUE_x >= Robservertotarget:
        return -1000

    # 计算方位角和仰角
    if use_relative_az_el:
        IMT_station = (0, 0)
        IMT_to_SUE_h, IMT_to_SUE_v = calculate_azimuth_elevation(
            IMT_station[0], IMT_station[1], BS_height,
            Satellite_UE[0], Satellite_UE[1], 1.5
        )

        # 使用相对方位角/仰角转换
        S_h_deg, S_v_deg = relative_az_el(
            degrees(IMT_to_SUE_h), degrees(IMT_to_SUE_v),
            90 - BS_tilt
        )
        S_h = radians(S_h_deg)
        S_v = radians(S_v_deg)

        # 计算接收增益
        rx_gain = calculate_rx_gain(
            temp_vic_config, 0, 0, S_v, S_h,
            az=degrees(S_h), el=degrees(S_v)
        )

        # 计算路径损耗
        pl = path_loss(IMT_to_SUE_x / 1000, band)

    elif is_ue_victim:
        # 用户终端作为受害方（scenario4）
        pl = path_loss(IMT_to_SUE_x / 1000, band)
        rx_gain = calculate_rx_gain(temp_vic_config, 0, 0, 0, 0)

    else:
        # 基站作为受害方（scenario1, 2, 3）
        IMT_station = (0, 0)
        IMT_UE = (UE_x * cos(_UE_h), UE_x * sin(_UE_h))

        IMT_to_SUE_h, IMT_to_SUE_v = calculate_azimuth_elevation(
            Satellite_UE[0], Satellite_UE[1], 1.5,
            IMT_station[0], IMT_station[1], BS_height
        )
        IMT_to_SUE_h = atan2(Satellite_UE[1], Satellite_UE[0])

        # 天线坐标转化
        UE_v = transformV(_UE_v, _UE_h, BS_tilt) - pi / 2
        UE_h = transformH(_UE_v, _UE_h, BS_tilt)
        S_v = transformV(IMT_to_SUE_v, IMT_to_SUE_h, BS_tilt)
        S_h = transformH(IMT_to_SUE_v, IMT_to_SUE_h, BS_tilt)

        # 计算接收增益
        rx_gain = calculate_rx_gain(temp_vic_config, UE_v, UE_h, S_v, S_h)

        # 计算路径损耗
        distance = sqrt(IMT_to_SUE_x**2 + (BS_height - BS_ue_height)**2) / 1000
        pl = path_loss(distance, band)

    # 计算发射增益（如果有自定义函数）
    if custom_tx_gain_func:
        A_A_out = custom_tx_gain_func()
    else:
        A_A_out = tx_gain

    # 计算干扰
    interference = tx_power + A_A_out + rx_gain - pl - ACLR - Body_loss + extra_gain_offset

    return interference


def singleInterference1_new():
    """
    IMT 基站与卫星系统的"挖洞"场景（新版本）
    计算保护区域范围，防止卫星终端干扰 IMT 基站

    使用配置：scenario1
    """
    return calculate_single_interference(
        scenario_name="scenario1",
        victim_distance_range=(0, None),  # 由 IMTUEInCell 决定
        satellite_distance_range=(isdis, Robservertotarget / 2),
        is_ue_victim=False
    )


def singleInterference2_new():
    """
    "拉远"场景（新版本）
    通过蒙特卡洛仿真评估高功率卫星终端对 IMT 基站的累积干扰概率分布

    使用配置：scenario2
    """
    return calculate_single_interference(
        scenario_name="scenario2",
        victim_distance_range=(0, None),
        satellite_distance_range=(0, Robservertotarget / 2),
        is_ue_victim=False
    )


def singleInterference_new():
    """
    HIBS 或远距离卫星系统对地面系统的干扰分析（新版本）
    受害方距离可达 100km

    使用配置：scenario3
    """
    return calculate_single_interference(
        int_scenario_name="scenario4",
        vic_scenario_name="scenario1",
        victim_distance_range=(0, 50000),
        satellite_distance_range=(0, Robservertotarget / 2),
        is_ue_victim=False,
        extra_gain_offset=10
    )


def singleInterference4_new():
    """
    IMT 用户终端受干扰场景（新版本）
    假设终端为全向天线，评估卫星对普通用户设备的干扰影响

    使用配置：scenario4
    """
    return calculate_single_interference(
        scenario_name="scenario4",
        victim_distance_range=(0, None),
        satellite_distance_range=(0, Robservertotarget / 2),
        is_ue_victim=True
    )


def singleInterference6_new():
    """
    固定业务或射电天文台等敏感系统保护场景（新版本）
    使用 ITU-R F.1336 标准天线模型进行合规性仿真

    使用配置：scenario6
    """
    return calculate_single_interference(
        scenario_name="scenario6",
        victim_distance_range=(0, Robservertotarget / 2),
        satellite_distance_range=(0, Robservertotarget / 2),
        use_relative_az_el=True
    )


def singleInterferenceOther_new():
    """
    射电天文、卫星气象等敏感系统干扰场景（新版本）

    使用配置：scenario_other
    """
    # 获取干扰站参数
    int_params = get_interference_params("scenario_other")
    tx_power = int_params["tx_power"]
    tx_gain_base = int_params["tx_gain"]

    # 获取受扰站参数
    vic_params = get_victim_params("scenario_other")

    # 受干扰方位置
    x = 0

    # 确定卫星 UE 的位置
    s_UE_h = random.uniform(-180, 180) / 180 * pi
    s_x = random.uniform(0, Robservertotarget / 2)
    Satellite_vector = [Satellite_dis + s_x * cos(s_UE_h), s_x * sin(s_UE_h), BS_ue_height]
    stationary = [-Satellite_dis, 0, BS_height - BS_ue_height]

    # 计算 IMT 到卫星 UE 的距离
    IMT_to_SUE_x = sqrt(Satellite_vector[0]**2 + Satellite_vector[1]**2)
    if IMT_to_SUE_x >= Robservertotarget:
        return -1000

    # 计算两个向量的夹角
    angle = calculate_angle(Satellite_vector, stationary)

    # 计算发射增益（使用 S580 模型）
    A_A_out = S580(angle)

    # 计算路径损耗
    pl = path_loss(sqrt(IMT_to_SUE_x**2 + (BS_height - BS_ue_height)**2) / 1000, band)

    # 计算干扰
    interference = tx_power + tx_gain_base + A_A_out - pl - ACLR - Body_loss

    return interference


def run_monte_carlo_simulation(scenario_name, num_iterations=1000):
    """
    运行蒙特卡洛仿真

    参数
    ----
    scenario_name : str
        场景名称
    num_iterations : int
        仿真迭代次数

    返回
    ----
    dict : 包含统计结果的字典
    """
    # 选择对应的仿真函数
    func_map = {
        "scenario1": singleInterference1_new,
        "scenario2": singleInterference2_new,
        "scenario3": singleInterference_new,
        "scenario4": singleInterference4_new,
        "scenario6": singleInterference6_new,
        "scenario_other": singleInterferenceOther_new
    }

    if scenario_name not in func_map:
        raise ValueError(f"未知场景：{scenario_name}")

    func = func_map[scenario_name]

    # 运行仿真
    results = []
    for _ in range(num_iterations):
        result = func()
        if result > -999:  # 过滤掉无效结果
            results.append(result)

    # 统计分析
    if len(results) == 0:
        return {"error": "No valid results"}

    results_np = np.array(results)

    return {
        "scenario": scenario_name,
        "iterations": len(results),
        "mean": float(np.mean(results_np)),
        "std": float(np.std(results_np)),
        "min": float(np.min(results_np)),
        "max": float(np.max(results_np)),
        "percentile_95": float(np.percentile(results_np, 95)),
        "percentile_99": float(np.percentile(results_np, 99))
    }


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("干扰仿真计算模块测试")
    print("=" * 60)

    # 显示所有可用的场景配置
    print("\n【干扰站配置】")
    for name, config in INTERFERENCE_STATION_CONFIGS.items():
        print(f"  {name}: {config['description']}")
        print(f"    发射功率：{config['tx_power']} dBm, 发射增益：{config['tx_gain']} dBi")

    print("\n【受扰站配置】")
    for name, config in VICTIM_STATION_CONFIGS.items():
        print(f"  {name}: {config['description']}")
        print(f"    天线模型：{config['antenna_model']}, 噪声系数：{config['noise_figure']} dB")

    # 运行单个场景测试
    print("\n【单场景测试】")
    test_scenario = "scenario4-scenario1"
    print(f"\n测试场景：{test_scenario}")
    try:
        result = singleInterference_new()
        print(f"  单次干扰计算结果：{result:.2f} dBm")
    except Exception as e:
        print(f"  错误：{e}")

    # 运行蒙特卡洛仿真
    print("\n【蒙特卡洛仿真】")
    sim_scenario = "scenario2"
    print(f"\n仿真场景：{sim_scenario}, 迭代次数：100")
    try:
        stats = run_monte_carlo_simulation(sim_scenario, num_iterations=100)
        print(f"  有效样本数：{stats['iterations']}")
        print(f"  平均值：{stats['mean']:.2f} dBm")
        print(f"  标准差：{stats['std']:.2f} dB")
        print(f"  最小值：{stats['min']:.2f} dBm")
        print(f"  最大值：{stats['max']:.2f} dBm")
        print(f"  95% 分位数：{stats['percentile_95']:.2f} dBm")
        print(f"  99% 分位数：{stats['percentile_99']:.2f} dBm")
    except Exception as e:
        print(f"  错误：{e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)