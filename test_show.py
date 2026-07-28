import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
# 创建随机极坐标数据
num_points = 10  # 点的数量
r = np.random.rand(num_points) * 5  # 随机半径，范围为 [0, 5]
theta = np.random.rand(num_points) * 2 * np.pi  # 随机角度，范围为 [0, 2π]

# 绘制极坐标散点图
plt.figure(figsize=(6, 6))
plt.polar(theta, r, 'o', label="数据点")  # 绘制散点
plt.title("极坐标散点图", fontsize=14)
plt.grid(True)

# 为每个点添加标签
for i in range(num_points):
    label = f"点{i + 1}"  # 标签内容
    # 调整标签位置，避免遮挡点
    offset = 0.2 # 偏移量
    plt.text(theta[i], r[i] + offset, label, ha='center', va='bottom', fontsize=8)

# 添加图例
plt.legend()

# 显示图形
plt.show()