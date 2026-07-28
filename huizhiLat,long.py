import matplotlib.pyplot as plt
import numpy as np

# 示例经纬度数据
latitudes = [45.000000, 47.600787, 47.600787,45.000000,42.399213,42.399213]
longitudes = [4.247068, 2.345871, -2.345871,-4.247068,-1.954397,1.954397]

# 创建一个简单的散点图
plt.scatter(longitudes, latitudes, color='blue', marker='o')

# 添加标题和标签
plt.title('经纬度示例')
plt.xlabel('经度')
plt.ylabel('纬度')

# 显示网格
plt.grid(True)

# 显示图形
plt.show()