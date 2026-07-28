import numpy as np
from matplotlib import pyplot as plt

from IMT_Simulation import Gain_TxF1336
from Parm_Simulation import BS_tilt, BS_radius, Robservertotarget, BS_height, range_SUE_num, SeparationDistance
from calculateInterference import singleInterference1, sumInterference
from show import CDF_plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Start, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Simulate_卫星终端对IMT基站干扰仿真')
    InterferenceList=[]
    # 绘制距离为x，干扰为y的散点图
    # d_km = np.arange(300, 509, 1)  # 单位 km
    # d_m = d_km * 1e3
    # Satellite_dis = d_m + Robservertotarget / 2
    # for tem in Satellite_dis:
    #     InterferenceList.append(sumInterference(range_SUE_num,tem))
    # InterferenceArray = np.array(InterferenceList)
    # plt.figure(figsize=(8, 5))
    # plt.plot(d_km,InterferenceList, lw=2)
    # # 标记 y = -115
    # plt.axhline(-115, color='red', ls='--', lw=1.5, label='-115 dBm')
    # # 可选：找到交点并垂直标记
    # idx = np.argmin(np.abs(InterferenceArray + 115))
    # x0, y0 = d_km[idx], InterferenceArray[idx]
    # plt.axvline(d_km[idx], color='red', ls='--', lw=1, label=f'{d_km[idx]} km')
    # # 标记交点
    # plt.scatter([x0], [y0], color='red', zorder=5)  # 红点
    # plt.text(x0 + 3, y0 - 1.5, f'({x0:.0f} km, {y0:.1f} dBm)',
    #          color='red', fontsize=9, va='top')
    # plt.title('MSS UE → HIBS Interference vs Separation Distance')
    # plt.xlabel('Separation Distance [km]')
    # plt.ylabel('Interference [dBm]')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # print(sumInterference(range_SUE_num))
    i=0
    for i in range(100):
        answer = sumInterference(range_SUE_num)
        # x=109#保护标准和I/N差值 IMT基站：109；IMT终端：106
        x=109+15
        if answer > -500:
            # print(++i)
            InterferenceList.append(answer+x) #绘制I/N的cdf图
    print(max(InterferenceList)-x)
    print(min(InterferenceList)-x)
    print(len(InterferenceList))
    CDF_plt(InterferenceList,"循环100次",SeparationDistance)
    # print(Gain_TxF1336(0,2.79))