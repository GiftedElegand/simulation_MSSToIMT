from math import sin, cos, atan, acos, log10

# 数学中的角度分弧度和角度两种，弧度的圆周是2π，而角度的圆周是360°
# 所以要将角度化为弧度就只需用角度乘以π/180，反之就除以(π/180)
from xml.etree.ElementTree import PI


def translateH(A_h, A_v, tilt):
    dt = tilt
    pi_fac = PI / 180
    mole = sin(A_h * pi_fac) * sin(A_v * pi_fac)
    deno_temp1 = cos(A_h * pi_fac) * sin(A_v * pi_fac) * cos(dt * pi_fac)
    deno_temp2 = cos(A_v * pi_fac) * sin(dt * pi_fac)
    Deg_H = atan(mole / (deno_temp1 - deno_temp2)) / pi_fac
    return Deg_H


def translateV(A_h, A_v, tilt):
    dt = tilt
    pi_fac = PI / 180
    a_temp = cos(A_h * pi_fac) * sin(A_v * pi_fac) * sin(dt * pi_fac)
    b_temp = cos(A_v * pi_fac) * cos(dt * pi_fac)
    Deg_V = acos(a_temp + b_temp) / pi_fac
    return Deg_V

# 计算接收增益
def Cal_Gain_tx_AAS(A_h, A_v, S_h, S_v):
    # 指向受扰站的方位角：-180~180
    # 指向受扰站的仰角：0~180
    # 指向终端的方位角：-180~180
    # 指向终端的仰角 - 90~90
    # Subarray AAS(extended AAS model)
    # AAS天线参数
    h_3dB = 90
    v_3dB = 65
    # 天线下倾角
    S_subtilt = 3
    Am = 30
    # 前后比
    SLV = 30
    # 元件增益
    Ge_Max = 6.4
    v_ra = 2.1
    h_ra = 0.5
    v_sub = 0.7
    M_row = 3
    row = 4
    column = 8
    # 单元阵列水平方向图
    G_h = 12 * (S_h / h_3dB) * (S_h / h_3dB)
    if G_h < Am:
        G_h = -1 * G_h
    else:
        G_h = -1 * Am

    G_v = 12 * ((S_v - 90) / v_3dB) * ((S_v - 90) / v_3dB)
    if G_v < SLV:
        G_v = -1 * G_v
    else:
        G_v = -1 * SLV

    G_A = -1 * (G_h + G_v)
    if G_A < Am:
        G_A = -1 * G_A
    else:
        G_A = -1 * Am

    G_A_E = Ge_Max + G_A
    w_m = v_sub * sin(S_subtilt * PI / 180)
    v_m = v_sub * cos(S_v * PI / 180)

    A_sub = 0
    A_sub_real = 0
    A_sub_image = 0
    for i in range(M_row):
        A_sub_real = A_sub_real + cos(2 * PI * (i - 1) * (w_m + v_m))
        A_sub_image = A_sub_image + sin(2 * PI * (i - 1) * (w_m + v_m))
    A_sub = G_A_E + 10 * log10((A_sub_real * A_sub_real + A_sub_image * A_sub_image) / M_row)
    w_v_m = v_ra * (cos(S_v * PI / 180) + sin(A_v * PI / 180))
    w_v_n = h_ra * (sin(S_v * PI / 180) * sin(S_h * PI / 180) - cos(A_v * PI / 180) * sin(A_h * PI / 180))
    A_A = 0
    A_A_real = 0
    A_A_image = 0
    for i in range(row):
        for j in range(column):
            A_A_real = A_A_real + cos(2 * PI * ((i - 1) * w_v_m + (j - 1) * w_v_n))
            A_A_image = A_A_image + sin(2 * PI * ((i - 1) * w_v_m + (j - 1) * w_v_n))
    A_A = A_sub + 10 * log10((A_A_real * A_A_real + A_A_image * A_A_image) / (row * column))
    return A_A

def Cal_Gain_tx_M2101(A_h, A_v, S_h, S_v, row, column):
    # 指向受扰站的方位角：-180~180
    # 指向受扰站的仰角：0~180
    # 指向终端的方位角：-180~180
    # 指向终端的仰角 - 90~90
    # Subarray AAS(extended AAS model)
    # AAS天线参数
    h_3dB = 65
    v_3dB = 65
    Am = 30
    # 前后比
    SLV = 30
    # 元件增益
    Ge_Max = 8
    v_ra = 0.5
    h_ra = 0.5

    # 单元阵列水平方向图
    G_h = 12 * (S_h / h_3dB) * (S_h / h_3dB)
    if G_h < Am:
        G_h = -1 * G_h
    else:
        G_h = -1 * Am

    G_v = 12 * ((S_v - 90) / v_3dB) * ((S_v - 90) / v_3dB)
    if G_v < SLV:
        G_v = -1 * G_v
    else:
        G_v = -1 * SLV

    G_A = -1 * (G_h + G_v)
    if G_A < Am:
        G_A = -1 * G_A
    else:
        G_A = -1 * Am


    G_A_E = Ge_Max + G_A

    w_v_m = v_ra * (cos(S_v * PI / 180) + sin(A_v * PI / 180))
    w_v_n = h_ra * (sin(S_v * PI / 180) * sin(S_h * PI / 180) - cos(A_v * PI / 180) * sin(A_h * PI / 180));
    A_A = 0
    A_A_real = 0
    A_A_image = 0
    for i in range(row):
        for j in range(column):
            A_A_real = A_A_real + cos(2 * PI * ((i - 1) * w_v_m + (j - 1) * w_v_n))
            A_A_image = A_A_image + sin(2 * PI * ((i - 1) * w_v_m + (j - 1) * w_v_n))
    A_A = G_A_E + 10 * log10((A_A_real * A_A_real + A_A_image * A_A_image) / (row * column))
    return A_A