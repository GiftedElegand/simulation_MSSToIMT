import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import sys
import os

# 确保可以导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from IMT_Simulation import path_loss
from Parm_Simulation import *

def calculate_single_interference(p_tx_mss, g_tx, g_rx, pl, body_loss, cable_loss):
    """
    计算单链路干扰
    Interference = MSS终端发射功率 + A_A_get + A_A_out - Ploss - Body_loss - closs
    """
    return p_tx_mss + g_tx + g_rx - pl - body_loss - cable_loss

def generate_mss_users(n_users, radius_km):
    """
    在半径为radius_km的圆内随机均匀生成MSS用户位置
    """
    # 极坐标生成均匀分布点
    r = radius_km * np.sqrt(np.random.uniform(0, 1, n_users))
    theta = np.random.uniform(0, 2 * np.pi, n_users)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # 假设MSS用户高度为1.5m (手持终端)
    z = np.ones(n_users) * 0.0015 
    return x, y, z

def calculate_aggregate_interference(mss_x, mss_y, mss_z, imt_x, imt_y, imt_z, params):
    """
    计算所有MSS用户对单个IMT用户的集总干扰
    """
    n_users = len(mss_x)
    interference_linear_sum = 0.0
    
    # 参数提取
    p_tx_mss = params['p_tx_mss']       # dBm
    body_loss = params['body_loss']     # dB
    cable_loss = params['cable_loss']   # dB
    freq = params['freq']               # GHz
    
    # 简化天线增益模型 (实际项目中应调用 calculate.py 中的详细角度计算)
    # 此处假设最坏情况或平均情况：
    # MSS终端指向卫星，IMT终端指向基站。
    # 为演示算法逻辑，假设两者相对方向的耦合增益为固定值或简单模型
    # 在实际集成中，此处应替换为 calculate_antenna_coupling_gain(lat1, lon1, lat2, lon2, ...)
    
    # 临时简化：假设主瓣对准增益为 G_max，旁瓣为 G_min
    # 这里为了体现距离变化对PL的影响，先假设增益为常数 (例如 0 dBi 全向或特定值)
    g_tx_avg = params.get('g_tx_avg', 0.0)  # MSS终端天线增益 dBi
    g_rx_avg = params.get('g_rx_avg', 0.0)  # IMT终端天线增益 dBi
    
    for i in range(n_users):
        # 计算距离 (km)
        dx = mss_x[i] - imt_x
        dy = mss_y[i] - imt_y
        dz = mss_z[i] - imt_z
        d_km = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if d_km < 0.001: # 避免除零或近场奇异点
            d_km = 0.001
            
        # 路径损耗 (dB) - 使用自由空间模型
        # IMT_Simulation.path_loss(DKM, M) -> M is Frequency in GHz? 
        # 检查原函数定义，通常 PL = 20log10(d) + 20log10(f) + 32.44
        pl = path_loss(d_km, freq)
        
        # 单用户干扰
        i_dbm = calculate_single_interference(
            p_tx_mss=p_tx_mss,
            g_tx=g_tx_avg,
            g_rx=g_rx_avg,
            pl=pl,
            body_loss=body_loss,
            cable_loss=cable_loss
        )
        
        # 线性累加
        interference_linear_sum += 10**(i_dbm / 10.0)
        
    # 转回 dBm
    if interference_linear_sum == 0:
        return -200.0 # 极低值
    return 10 * np.log10(interference_linear_sum)

def run_optimization_simulation():
    print("=== MSS UE to IMT UE Interference Distance Optimization ===")
    
    # 1. 场景参数设置
    R_circle = 8.7  # 圆形区域半径 km
    N_users = 100   # MSS用户数量
    step_d = 0.1    # 移动步长 km
    max_iter = 1000 # 最大迭代次数防止死循环
    
    # 2. 系统参数 (参考 Parm_Simulation 或自定义)
    # 注意：需要从 Parm_Simulation 中获取真实参数，这里做示例性填充
    # 实际使用时请根据 Parm_Simulation.py 中的具体变量名调整
    params = {
        'p_tx_mss': 23.0,      # MSS终端发射功率 dBm (假设)
        'freq': 2.0,           # 频率 GHz (假设 S波段或C波段，需与实际一致)
        'body_loss': 3.0,      # 人体损耗 dB
        'cable_loss': 0.0,     # 电缆损耗 dB (终端通常无外接电缆，设为0或连接器损耗)
        'g_tx_avg': -35.0,     # MSS天线增益 dBi (考虑指向卫星，对地面IMT为后瓣/旁瓣，典型值-30~-40dBi)
        'g_rx_avg': -35.0,     # IMT天线增益 dBi (考虑指向基站，对MSS用户为后瓣/旁瓣，典型值-30~-40dBi)
        
        # 噪声参数
        'bw_mhz': 10.0,        # 带宽 MHz
        'nf_db': 5.0           # 噪声系数 dB
    }
    
    # 更新频率参数以匹配项目实际 (如果 IMT_Simulation.path_loss 需要特定单位)
    # 假设 path_loss(DKM, M) 中 M 是 GHz
    
    # 3. 生成MSS用户位置 (固定种子以保证可复现性，可选)
    np.random.seed(42)
    mss_x, mss_y, mss_z = generate_mss_users(N_users, R_circle)
    
    # 4. IMT终端初始位置
    imt_x = R_circle
    imt_y = 0.0
    imt_z = 0.0015 # 1.5m
    
    # 5. 计算噪声基底
    # N = -174 + 10*log10(BW_Hz) + NF
    bw_hz = params['bw_mhz'] * 1e6
    noise_floor = -174.0 + 10 * np.log10(bw_hz) + params['nf_db']
    threshold_in = noise_floor - 6.0 # I/N <= -6 dB
    
    print(f"噪声基底: {noise_floor:.2f} dBm")
    print(f"干扰门限 (I/N <= -6): {threshold_in:.2f} dBm")
    print(f"初始IMT位置: ({imt_x:.2f}, {imt_y:.2f}) km")
    
    current_d = 0.0
    iteration = 0
    
    history_d = []
    history_i = []
    
    while iteration < max_iter:
        # 计算当前集总干扰
        agg_interference = calculate_aggregate_interference(
            mss_x, mss_y, mss_z,
            imt_x, imt_y, imt_z,
            params
        )
        
        history_d.append(current_d)
        history_i.append(agg_interference)
        
        print(f"Iter {iteration}: d={current_d:.2f} km, Pos=({imt_x:.2f}, 0), I_total={agg_interference:.2f} dBm, I/N={agg_interference - noise_floor:.2f} dB")
        
        if agg_interference <= threshold_in:
            print(f"\n>>> 满足保护标准！")
            break
        
        # 不满足，移动IMT终端
        current_d += step_d
        imt_x = R_circle + current_d
        iteration += 1
        
    if iteration == max_iter:
        print("警告：达到最大迭代次数，可能未完全收敛或场景过于恶劣。")
        
    final_distance = current_d
    final_interference = history_i[-1]
    
    print(f"\n=== 结果汇总 ===")
    print(f"所需最小移动距离 d: {final_distance:.2f} km")
    print(f"最终IMT位置: ({R_circle + final_distance:.2f}, 0) km")
    print(f"最终集总干扰: {final_interference:.2f} dBm")
    print(f"最终 I/N: {final_interference - noise_floor:.2f} dB")
    
    # 6. 绘制CDF图
    # 为了画CDF，我们需要在最终位置进行一次蒙特卡洛采样（因为之前的循环只算了一个样本点）
    # 或者，如果认为用户分布是随机的，我们应该在最终位置重新运行多次用户分布的生成来计算统计特性
    # 这里为了展示“在此处的CDF”，我们保持用户分布不变（确定性分析），
    # 但通常干扰分析需要统计特性。
    # **修正策略**：上面的循环是基于一次随机撒点的确定性距离寻找。
    # 若要画CDF，通常意味着用户位置也是随机变量。
    # 既然题目说“随机撒点N个”，这通常指一次实现。
    # 如果要画CDF，合理的解释是：在确定的最终距离d处，考虑MSS用户位置的随机性（重新撒点多次）。
    
    print(f"\n正在生成最终位置下的干扰CDF分布 (基于1000次MSS用户随机分布)...")
    
    cdf_samples = []
    n_monte_carlo_runs = 1000
    
    fixed_imt_x_final = R_circle + final_distance
    fixed_imt_y_final = 0.0
    fixed_imt_z_final = 0.0015
    
    for _ in range(n_monte_carlo_runs):
        # 重新撒点
        mx, my, mz = generate_mss_users(N_users, R_circle)
        i_val = calculate_aggregate_interference(
            mx, my, mz,
            fixed_imt_x_final, fixed_imt_y_final, fixed_imt_z_final,
            params
        )
        cdf_samples.append(i_val)
        
    cdf_samples = np.array(cdf_samples)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    sorted_samples = np.sort(cdf_samples)
    cdf_values = np.arange(1, len(sorted_samples)+1) / len(sorted_samples)
    
    plt.plot(sorted_samples, cdf_values, linewidth=2, color='b')
    plt.axvline(x=threshold_in, color='r', linestyle='--', label=f'Threshold (I/N=-6dB): {threshold_in:.1f} dBm')
    plt.axvline(x=np.mean(cdf_samples), color='g', linestyle=':', label=f'Mean: {np.mean(cdf_samples):.1f} dBm')
    
    plt.title(f'CDF of Aggregate Interference at d={final_distance:.2f} km\n(IMT Position: {fixed_imt_x_final:.2f}, 0) km')
    plt.xlabel('Aggregate Interference Power (dBm)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    save_path = 'Interference_CDF_Optimized_Distance.png'
    plt.savefig(save_path)
    print(f"CDF图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_optimization_simulation()
