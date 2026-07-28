import math
from fontTools.misc.py23 import range

# 地球半径（单位：米）
R = 6371000

# 卫星参数
sat_lat = 45.0  # 卫星纬度（单位：度）
sat_lon = 0.0  # 卫星经度（单位：度）
sat_height = 35786000  # 卫星高度（单位：米，例如GEO卫星的高度）

# 波束参数
beam_width_deg = 5.0  # 波束宽度（单位：度）
num_beams = 7  # 波束数量

# 将角度转换为弧度
beam_width_rad = math.radians(beam_width_deg)

# 计算波束在地表的覆盖半径（简化方法，未考虑地球曲率）
# 注意：这个公式在卫星高度远大于地球半径时可能不准确
r_beam = R * math.tan(beam_width_rad / 2) * (sat_height / (R + sat_height))

# 计算波束中心点
beam_centers = []

for i in range(num_beams-1):
    # 计算波束与正北方向的夹角（单位：弧度）
    theta = 2 * math.pi * i / (num_beams-1)

    # 计算纬度变化（简化方法，未考虑地球曲率）
    # 注意：这个公式在卫星纬度较高时可能不准确
    delta_lat = r_beam / R * (1 / math.cos(math.radians(sat_lat)))
    lat_center = sat_lat + math.degrees(delta_lat * math.sin(theta))

    # 由于经度变化与纬度有关，我们在这里做一个简化的近似
    # 注意：这个公式在波束覆盖区域较大或卫星纬度较高时可能不准确
    delta_lon = r_beam / R * math.cos(math.radians(sat_lat)) / math.cos(math.radians(lat_center))
    lon_center = sat_lon + math.degrees(delta_lon * math.cos(theta))

    # 将中心点添加到列表中
    beam_centers.append((lat_center, lon_center))

# 打印波束中心点经纬度
for center in beam_centers:
    print(f"Beam center: Latitude = {center[0]:.6f}, Longitude = {center[1]:.6f}")

