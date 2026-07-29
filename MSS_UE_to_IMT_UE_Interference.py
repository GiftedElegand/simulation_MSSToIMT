import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False
# ================= 参数配置（同前） =================
NUM_MSS_USERS = 1
R_AREA = 8.7
FREQ_GHZ = 2.0
NOISE_DENSITY_DBM = -111.0
BANDWIDTH_MHZ = 1.0
NOISE_FLOOR = NOISE_DENSITY_DBM + 10 * np.log10(BANDWIDTH_MHZ)
TARGET_IN_RATIO_DB = -6.0
MAX_ITERATIONS = 500
STEP_SIZE_KM_INIT = 0.8
STEP_DECAY = 0.98

MSS_TX_POWER_DBM = 33.4
MSS_ANT_GAIN_DB = -0
IMT_RX_ANT_GAIN_DB = -3.0
BODY_LOSS_DB = 4.0
CABLE_LOSS_DB = 0.0

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ================= 传播模型（同前） =================
def free_space_path_loss(d_km, f_ghz):
    f_mhz = f_ghz * 1000
    if d_km < 0.001:
        d_km = 0.001
    return 32.44 + 20 * np.log10(d_km) + 20 * np.log10(f_mhz)

def clutter_loss(f_ghz, p, d_km):
    return 10 + 0.5 * f_ghz + 2 * np.log10(max(d_km, 0.01))

def calculate_single_interference(dist_km):
    pl = free_space_path_loss(dist_km, FREQ_GHZ)
    cl = clutter_loss(FREQ_GHZ, 0.5, dist_km)
    return (MSS_TX_POWER_DBM + MSS_ANT_GAIN_DB + IMT_RX_ANT_GAIN_DB
            - pl - cl - BODY_LOSS_DB - CABLE_LOSS_DB)

# ================= 评估函数 =================
def evaluate_position(imt_x, imt_y, n_samples=1000):
    """
    在给定 IMT 位置，生成 n_samples 组随机 MSS 用户，
    返回每个样本的聚合干扰 I/N (dB)
    """
    in_ratios = []
    for _ in range(n_samples):
        # 生成一组 MSS 用户位置
        angles = np.random.uniform(0, 2*np.pi, NUM_MSS_USERS)
        radii = np.sqrt(np.random.uniform(0, 1, NUM_MSS_USERS)) * R_AREA
        mss_x = radii * np.cos(angles)
        mss_y = radii * np.sin(angles)
        # 计算距离
        dists = np.sqrt((mss_x - imt_x)**2 + (mss_y - imt_y)**2)
        dists = np.maximum(dists, 0.001)
        # 计算每个用户干扰
        inters = [calculate_single_interference(d) for d in dists]
        # 聚合
        if NUM_MSS_USERS == 1:
            total_interf = inters[0]
        else:
            total_interf = 10 * np.log10(np.sum(10 ** (np.array(inters)/10.0)))
        in_ratio = total_interf - NOISE_FLOOR
        in_ratios.append(in_ratio)
    return np.array(in_ratios)

# ================= 主仿真（鲁棒版） =================
def run_simulation_robust():
    print("=== MSS -> IMT 干扰仿真 (鲁棒判据：1000个样本全部达标) ===")
    print(f"MSS用户数: {NUM_MSS_USERS}, 区域半径: {R_AREA} km")
    print(f"噪声基底: {NOISE_FLOOR:.2f} dBm, 目标 I/N: {TARGET_IN_RATIO_DB} dB")

    # 初始 IMT 位置
    imt_x = R_AREA
    imt_y = 0.0
    trajectory = [(imt_x, imt_y)]

    step_size = STEP_SIZE_KM_INIT
    best_result = None   # 记录最接近达标的位置 (以防达到最大迭代)

    for i in range(MAX_ITERATIONS):
        # 评估当前位置的 1000 个样本
        in_ratios = evaluate_position(imt_x, imt_y, n_samples=1000)

        # 判断是否全部达标
        all_pass = np.all(in_ratios <= TARGET_IN_RATIO_DB)
        max_in = np.max(in_ratios)

        if i % 20 == 0:
            print(f"Iter {i}: pos=({imt_x:.2f}, {imt_y:.2f}), "
                  f"max I/N={max_in:.2f} dB, pass rate={np.mean(in_ratios <= TARGET_IN_RATIO_DB)*100:.1f}%")

        if all_pass:
            print(f"\n达标! 迭代次数: {i+1}, 移动距离: {np.sqrt((imt_x-R_AREA)**2 + imt_y**2):.2f} km")
            return imt_x, imt_y, in_ratios, trajectory

        # 保存当前最佳 (如果达标率更高)
        if best_result is None or np.mean(in_ratios <= TARGET_IN_RATIO_DB) > np.mean(best_result[2] <= TARGET_IN_RATIO_DB):
            best_result = (imt_x, imt_y, in_ratios)

        # ---- 移动策略 ----
        # 为了移动方向，我们依然使用当前固定采样的一组 MSS 位置？为了稳定性，我们可以取最后评估的一组位置来求质心
        angles_fixed = np.random.uniform(0, 2*np.pi, NUM_MSS_USERS)
        radii_fixed = np.sqrt(np.random.uniform(0, 1, NUM_MSS_USERS)) * R_AREA
        mss_x_fixed = radii_fixed * np.cos(angles_fixed)
        mss_y_fixed = radii_fixed * np.sin(angles_fixed)
        centroid_x = np.mean(mss_x_fixed)
        centroid_y = np.mean(mss_y_fixed)
        dir_x = imt_x - centroid_x
        dir_y = imt_y - centroid_y
        norm = np.sqrt(dir_x**2 + dir_y**2)
        if norm < 1e-6:
            dir_x, dir_y = 1.0, 0.0
        else:
            dir_x /= norm
            dir_y /= norm

        imt_x += step_size * dir_x
        imt_y += step_size * dir_y
        # 边界限制
        max_bound = 10 * R_AREA
        imt_x = np.clip(imt_x, -max_bound, max_bound)
        imt_y = np.clip(imt_y, -max_bound, max_bound)

        trajectory.append((imt_x, imt_y))
        step_size *= STEP_DECAY
        if step_size < 0.01:
            step_size = 0.01

    # 未完全达标，输出最佳结果
    print("\n警告: 达到最大迭代次数，未找到100%达标位置，输出最佳结果（最高达标率）")
    bx, by, best_in = best_result
    return bx, by, best_in, trajectory

# ================= 绘图函数 =================
def plot_results_robust(trajectory, in_ratios_final):
    """
    绘制轨迹和 I/N 的 CDF (此时所有点应 <= -6dB)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 轨迹
    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]
    ax1.plot(traj_x, traj_y, 'o-', markersize=4, linewidth=1.5, label='IMT trajectory')
    ax1.scatter([traj_x[0]], [traj_y[0]], color='green', s=80, label='Start')
    ax1.scatter([traj_x[-1]], [traj_y[-1]], color='red', s=80, label='End')
    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('y (km)')
    ax1.set_title('IMT 移动轨迹')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')

    # CDF of I/N
    sorted_in = np.sort(in_ratios_final)
    cdf = np.arange(1, len(sorted_in)+1) / len(sorted_in)
    ax2.plot(sorted_in, cdf, linewidth=2, label='I/N CDF')


    ax2.axvline(x=-6.0, color='g', linestyle=':', linewidth=2, label='I/N = -6 dB')
    ax2.text(-6.0, 0.5, 'I/N = -6 dB', rotation=90, verticalalignment='center',
             color='g', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax2.set_xlabel('I/N (dB)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title(f'干扰 I/N 的 CDF (100% 样本达标)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('MSS_IMT_Robust_CDF.png', dpi=150)
    plt.show()

# ================= 主程序 =================
if __name__ == "__main__":
    # 运行鲁棒仿真
    final_x, final_y, in_ratios, traj = run_simulation_robust()
    # 此时 in_ratios 已经是 1000 个样本，且全部达标
    print(f"\n最终位置: ({final_x:.2f}, {final_y:.2f}) km")
    print(f"移动距离: {np.sqrt((final_x-R_AREA)**2 + final_y**2):.2f} km")
    print(f"达标样本数: {np.sum(in_ratios <= -6.0)} / {len(in_ratios)}")

    # 绘制结果
    plot_results_robust(traj, in_ratios)