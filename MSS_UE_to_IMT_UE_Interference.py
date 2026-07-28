"""
MSS终端对IMT终端干扰仿真算法
================================
应用场景：评估卫星直连终端（MSS UE）对地面蜂窝用户（IMT UE）的干扰
核心特性：
    1. 双移动性建模：干扰源和受害体均为移动终端
    2. 自由空间路径损耗：直接复用 IMT_Simulation.path_loss
    3. 3D几何耦合：计算终端间相对方位角/仰角
    4. 聚合干扰统计：支持多MSS用户对单IMT用户的干扰叠加
"""

import numpy as np
import math
from math import log10, sqrt, atan2, degrees, radians
from IMT_Simulation import path_loss, gain_calc, relative_az_el, vec, az_el_from_vec
from Parm_Simulation import BS_ue_height  # 假设IMT UE高度已定义

# ======================== 参数配置 ========================
class InterferenceConfig:
    """干扰仿真配置参数"""
    # 频率配置 (MHz)
    FREQ_MHZ = 2000  # 2 GHz
    
    # MSS终端参数
    MSS_TX_POWER_DBM = 23  # 23 dBm (0.2W)
    MSS_ANT_HEIGHT_M = 1.5  # 手持终端高度
    MSS_ANT_GAIN_MAX = 5   # dBi (全向天线近似)
    
    # IMT终端参数
    IMT_UE_HEIGHT_M = 1.5  # 手机用户高度
    IMT_RX_BANDWIDTH_MHZ = 10  # 接收带宽
    
    # 仿真参数
    NUM_MSS_USERS = 50  # 同时干扰的MSS用户数
    SIMULATION_AREA_M = 5000  # 仿真区域半径 (米)


# ======================== 几何计算模块 ========================
def calculate_3d_distance(ue1_pos, ue2_pos):
    """
    计算两个终端之间的3D欧氏距离
    
    参数:
        ue1_pos: (lon, lat, alt) 或 (x, y, z) [米]
        ue2_pos: (lon, lat, alt) 或 (x, y, z) [米]
    
    返回:
        distance_m: 距离 [米]
        distance_km: 距离 [千米]
    """
    dx = ue1_pos[0] - ue2_pos[0]
    dy = ue1_pos[1] - ue2_pos[1]
    dz = ue1_pos[2] - ue2_pos[2]
    
    distance_m = sqrt(dx**2 + dy**2 + dz**2)
    distance_km = distance_m / 1000.0
    
    return distance_m, distance_km


def calculate_relative_angles(mss_pos, imt_pos, mss_azimuth, mss_elevation):
    """
    计算MSS终端天线主瓣方向与IMT终端方向的相对角度
    
    参数:
        mss_pos: MSS终端位置 (x, y, z)
        imt_pos: IMT终端位置 (x, y, z)
        mss_azimuth: MSS终端指向卫星的方位角 [度]
        mss_elevation: MSS终端指向卫星的仰角 [度]
    
    返回:
        rel_az: 相对方位角 [度]
        rel_el: 相对仰角 [度]
        coupling_angle: 耦合夹角 [度]
    """
    # 计算MSS到IMT的方向向量
    dx = imt_pos[0] - mss_pos[0]
    dy = imt_pos[1] - mss_pos[1]
    dz = imt_pos[2] - mss_pos[2]
    
    # 转换为方位角和仰角
    az_mss_to_imt = degrees(atan2(dy, dx))
    el_mss_to_imt = degrees(atan2(dz, sqrt(dx**2 + dy**2)))
    
    # 使用relative_az_el计算相对角度（考虑MSS指向卫星的波束方向）
    rel_az, rel_el = relative_az_el(
        az_a=az_mss_to_imt, 
        el_a=el_mss_to_imt,
        az_b=mss_azimuth,
        el_b=mss_elevation
    )
    
    # 计算三维空间中的耦合夹角
    v_main = vec(mss_azimuth, mss_elevation)  # 主瓣方向向量
    v_interf = vec(az_mss_to_imt, el_mss_to_imt)  # 干扰方向向量
    
    dot_product = np.dot(v_main, v_interf)
    coupling_angle = degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
    
    return rel_az, rel_el, coupling_angle


# ======================== 天线增益模块 ========================
def calculate_coupling_gain(rel_az, rel_el, coupling_angle):
    """
    计算MSS终端天线在IMT终端方向上的耦合增益
    
    参数:
        rel_az: 相对方位角 [度]
        rel_el: 相对仰角 [度]
        coupling_angle: 耦合夹角 [度]
    
    返回:
        gain_dB: 耦合增益 [dB]
    """
    # 方法1: 使用简化的余弦模型（适用于全向/半全向天线）
    # G = G_max * cos^n(theta)，n通常取1-2
    G_max = InterferenceConfig.MSS_ANT_GAIN_MAX
    
    # 将角度转换为弧度
    theta_rad = radians(coupling_angle)
    
    # 简化模型：G = G_max - 12*(theta/theta_3dB)^2
    theta_3dB = 65  # 3dB波束宽度 [度]
    if coupling_angle <= 90:
        gain_dB = G_max - 12 * (coupling_angle / theta_3dB)**2
    else:
        gain_dB = -10  # 后瓣增益
    
    # 方法2: 如果需要更精确的AAS模型，可以调用gain_calc
    # 注意：需要将相对角度转换为gain_calc所需的格式
    # gain_dB = gain_calc(UE_v=radians(rel_el), UE_h=radians(rel_az), 
    #                     S_v=radians(0), S_h=radians(0))
    
    return gain_dB


# ======================== 干扰计算核心 ========================
def calculate_single_interference(mss_pos, imt_pos, mss_az, mss_el):
    """
    计算单个MSS终端对单个IMT终端的干扰功率
    
    参数:
        mss_pos: MSS终端位置 (x, y, z) [米]
        imt_pos: IMT终端位置 (x, y, z) [米]
        mss_az: MSS终端指向卫星的方位角 [度]
        mss_el: MSS终端指向卫星的仰角 [度]
    
    返回:
        interference_dbm: 干扰功率 [dBm]
        path_loss_db: 路径损耗 [dB]
        coupling_gain_db: 耦合增益 [dB]
    """
    # 1. 计算距离
    dist_m, dist_km = calculate_3d_distance(mss_pos, imt_pos)
    
    # 避免除零错误
    if dist_km < 0.001:
        dist_km = 0.001
    
    # 2. 计算自由空间路径损耗 (调用IMT_Simulation.path_loss)
    pl_db = path_loss(dist_km, InterferenceConfig.FREQ_MHZ)
    
    # 3. 计算相对角度和耦合增益
    rel_az, rel_el, coupling_angle = calculate_relative_angles(
        mss_pos, imt_pos, mss_az, mss_el
    )
    coupling_gain_db = calculate_coupling_gain(rel_az, rel_el, coupling_angle)
    
    # 4. 计算干扰功率
    # I = Tx_Power + G_coupling - Path_Loss
    interference_dbm = (InterferenceConfig.MSS_TX_POWER_DBM + 
                        coupling_gain_db - pl_db)
    
    return interference_dbm, pl_db, coupling_gain_db


def calculate_aggregate_interference(imt_pos, num_mss_users=InterferenceConfig.NUM_MSS_USERS, 
                                     simulation_radius=InterferenceConfig.SIMULATION_AREA_M):
    """
    计算多个MSS终端对单个IMT终端的聚合干扰
    
    参数:
        imt_pos: IMT终端位置 (x, y, z) [米]
        num_mss_users: MSS用户数量
        simulation_radius: 仿真区域半径 [米]
    
    返回:
        total_interference_dbm: 总干扰功率 [dBm]
        interference_list: 各MSS用户的干扰功率列表 [dBm]
        stats: 统计信息字典
    """
    interference_list = []
    
    # 随机生成MSS用户位置和指向
    for i in range(num_mss_users):
        # 随机位置（极坐标分布，保证均匀分布）
        r = np.random.uniform(0, simulation_radius)
        theta = np.random.uniform(0, 2 * math.pi)
        
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = InterferenceConfig.MSS_ANT_HEIGHT_M
        
        mss_pos = (x, y, z)
        
        # 随机指向（假设MSS终端指向天顶附近的卫星）
        mss_az = np.random.uniform(-180, 180)
        mss_el = np.random.uniform(30, 90)  # 仰角通常较高
        
        # 计算单个干扰
        intf_dbm, pl_db, cg_db = calculate_single_interference(
            mss_pos, imt_pos, mss_az, mss_el
        )
        interference_list.append(intf_dbm)
    
    # 线性域叠加
    interference_linear = [10**(i/10) for i in interference_list]
    total_interference_linear = sum(interference_linear)
    total_interference_dbm = 10 * log10(total_interference_linear)
    
    # 统计信息
    stats = {
        'mean_interference_dbm': np.mean(interference_list),
        'std_interference_dbm': np.std(interference_list),
        'max_interference_dbm': np.max(interference_list),
        'min_interference_dbm': np.min(interference_list),
        'num_users': num_mss_users
    }
    
    return total_interference_dbm, interference_list, stats


# ======================== SIR评估模块 ========================
def calculate_sir(imt_pos, desired_signal_dbm, num_mss_users=InterferenceConfig.NUM_MSS_USERS):
    """
    计算信干比 (SIR)
    
    参数:
        imt_pos: IMT终端位置
        desired_signal_dbm: 期望信号功率（来自IMT基站）[dBm]
        num_mss_users: 干扰的MSS用户数
    
    返回:
        sir_db: 信干比 [dB]
        interference_dbm: 总干扰功率 [dBm]
    """
    total_interference_dbm, _, _ = calculate_aggregate_interference(
        imt_pos, num_mss_users
    )
    
    # SIR = S - I (dB域)
    sir_db = desired_signal_dbm - total_interference_dbm
    
    return sir_db, total_interference_dbm


# ======================== 蒙特卡洛仿真 ========================
def run_monte_carlo_simulation(num_iterations=1000, num_mss_users=InterferenceConfig.NUM_MSS_USERS):
    """
    运行蒙特卡洛仿真，生成干扰统计分布
    
    参数:
        num_iterations: 仿真迭代次数
        num_mss_users: 每次仿真的MSS用户数
    
    返回:
        results: 包含所有仿真结果的字典
    """
    interference_results = []
    sir_results = []
    
    # 假设IMT UE固定在原点
    imt_pos = (0, 0, InterferenceConfig.IMT_UE_HEIGHT_M)
    
    # 假设置望信号功率（根据3GPP UMi模型估算）
    # 假设IMT UE距离基站200米
    desired_signal_power = -70  # dBm (典型值)
    
    print(f"开始蒙特卡洛仿真，迭代次数：{num_iterations}")
    print(f"MSS用户数：{num_mss_users}, 频率：{InterferenceConfig.FREQ_MHZ} MHz")
    print("-" * 60)
    
    for i in range(num_iterations):
        # 计算聚合干扰
        total_intf, intf_list, stats = calculate_aggregate_interference(
            imt_pos, num_mss_users
        )
        interference_results.append(total_intf)
        
        # 计算SIR
        sir_db, _ = calculate_sir(imt_pos, desired_signal_power, num_mss_users)
        sir_results.append(sir_db)
        
        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"进度：{i+1}/{num_iterations}, "
                  f"平均干扰：{np.mean(interference_results):.2f} dBm, "
                  f"平均SIR：{np.mean(sir_results):.2f} dB")
    
    # 统计分析
    results = {
        'interference_mean_dbm': np.mean(interference_results),
        'interference_std_dbm': np.std(interference_results),
        'interference_percentiles': np.percentile(interference_results, [50, 90, 95, 99]),
        'sir_mean_db': np.mean(sir_results),
        'sir_std_db': np.std(sir_results),
        'sir_percentiles': np.percentile(sir_results, [1, 5, 10, 50]),
        'raw_interference': interference_results,
        'raw_sir': sir_results
    }
    
    print("\n" + "=" * 60)
    print("仿真结果汇总:")
    print(f"  平均干扰功率：{results['interference_mean_dbm']:.2f} dBm")
    print(f"  干扰标准差：{results['interference_std_dbm']:.2f} dB")
    print(f"  干扰中位数：{results['interference_percentiles'][0]:.2f} dBm")
    print(f"  干扰95%分位：{results['interference_percentiles'][2]:.2f} dBm")
    print(f"  平均SIR：{results['sir_mean_db']:.2f} dB")
    print(f"  SIR 1%分位（最坏情况）：{results['sir_percentiles'][0]:.2f} dB")
    print("=" * 60)
    
    return results


# ======================== 可视化辅助 ========================
def plot_cdf(data, title="CDF曲线", xlabel="值", ylabel="累积概率"):
    """绘制CDF曲线"""
    try:
        import matplotlib.pyplot as plt
        
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_data, cdf, linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib未安装，跳过绘图")


# ======================== 主函数示例 ========================
if __name__ == "__main__":
    print("=" * 60)
    print("MSS终端对IMT终端干扰仿真系统")
    print("=" * 60)
    
    # 运行蒙特卡洛仿真
    results = run_monte_carlo_simulation(num_iterations=500, num_mss_users=50)
    
    # 绘制CDF曲线
    print("\n生成干扰功率CDF曲线...")
    plot_cdf(results['raw_interference'], 
             title="MSS→IMT UE干扰功率CDF分布",
             xlabel="干扰功率 [dBm]", 
             ylabel="累积概率")
    
    print("\n生成SIR CDF曲线...")
    plot_cdf(results['raw_sir'], 
             title="IMT UE信干比(SIR)CDF分布",
             xlabel="SIR [dB]", 
             ylabel="累积概率")
    
    print("\n仿真完成！")
