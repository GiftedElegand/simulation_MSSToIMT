import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def read_data_from_file(filename):
    """从文本文件读取数据"""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # 跳过空行和注释行
            if line.strip() == '' or line.startswith('#'):
                continue
            try:
                # 尝试转换为浮点数
                data.append(float(line.strip()))
            except ValueError:
                # 跳过无法转换的行
                continue
    return np.array(data)


def plot_cdf(data, output_file=None, title='Cumulative Distribution Function'):
    """绘制CDF曲线"""
    # 对数据进行排序
    sorted_data = np.sort(data)

    # 计算CDF值 (使用 (i+1)/N 而不是 i/N 避免0%的问题)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, cdf, 'b-', linewidth=2)

    # 添加标签和标题
    plt.title(title, fontsize=14)
    plt.xlabel('I/N(dB)', fontsize=12)
    plt.ylabel('Probability(%)', fontsize=12)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置Y轴范围
    plt.ylim(0, 1)

    # 标记x=-6的点并显示其概率
    x_marker = -6.0

    # 找到最接近x=-6的数据点
    idx = np.searchsorted(sorted_data, x_marker)

    # 确保索引在有效范围内
    if idx >= len(sorted_data):
        idx = len(sorted_data) - 1

    # 计算x=-6处的概率
    if x_marker < sorted_data[0]:
        prob_at_x = 0.0
    elif x_marker > sorted_data[-1]:
        prob_at_x = 1.0
    else:
        # 使用线性插值估算x=-6处的概率
        if idx == 0:
            prob_at_x = cdf[0]
        else:
            # 计算插值权重
            weight = (x_marker - sorted_data[idx - 1]) / (sorted_data[idx] - sorted_data[idx - 1])
            prob_at_x = cdf[idx - 1] + weight * (cdf[idx] - cdf[idx - 1])

    # 绘制标记点
    plt.plot(x_marker, prob_at_x, 'ro', markersize=8, label=f'x={x_marker}, P(X≤x)={prob_at_x:.3f}')

    # 添加指向标记点的注释
    plt.annotate(f'P(X≤{x_marker}) = {prob_at_x:.3f}',
                 xy=(x_marker, prob_at_x),
                 xytext=(x_marker + 0.5, prob_at_x + 0.1),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

    # 添加垂直线和水平线以帮助定位
    plt.axvline(x=x_marker, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=prob_at_x, color='red', linestyle='--', alpha=0.5)

    # 标记80%概率对应的点
    prob_80 = 0.8
    # 找到最接近80%概率的数据点
    idx_80 = np.searchsorted(cdf, prob_80)

    # 确保索引在有效范围内
    if idx_80 >= len(sorted_data):
        idx_80 = len(sorted_data) - 1

    # 计算80%概率对应的x值
    if idx_80 == 0:
        x_at_80 = sorted_data[0]
    else:
        # 使用线性插值估算x值
        weight = (prob_80 - cdf[idx_80 - 1]) / (cdf[idx_80] - cdf[idx_80 - 1])
        x_at_80 = sorted_data[idx_80 - 1] + weight * (sorted_data[idx_80] - sorted_data[idx_80 - 1])

    # 绘制80%标记点
    plt.plot(x_at_80, prob_80, 'go', markersize=8, label=f'P(X≤x)={prob_80}, x={x_at_80:.3f}')

    # 添加指向80%标记点的注释
    plt.annotate(f'P(X≤x) = {prob_80}, x={x_at_80:.3f}',
                 xy=(x_at_80, prob_80),
                 xytext=(x_at_80 + 0.5, prob_80 - 0.1),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))

    # 添加垂直线和水平线以帮助定位80%点
    plt.axvline(x=x_at_80, color='green', linestyle='--', alpha=0.5)
    plt.axhline(y=prob_80, color='green', linestyle='--', alpha=0.5)

    # 添加图例
    plt.legend(loc='lower right')

    # 添加数据统计信息
    # stats_text = f"Data Points: {len(data)}\nMin: {np.min(data):.4f}\nMax: {np.max(data):.4f}\nMean: {np.mean(data):.4f}\nStd Dev: {np.std(data):.4f}"
    # plt.annotate(stats_text, xy=(0.98, 0.02), xycoords='axes fraction',
    #              ha='right', va='bottom', fontsize=10,
    #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # 保存或显示图像
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"CDF plot saved to: {os.path.abspath(output_file)}")
    else:
        plt.show()


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Plot CDF from data in a text file.')
    parser.add_argument('input_file', help='Input text file with data points (one per line)')
    parser.add_argument('-o', '--output', help='Output image file (e.g., plot.png)', default=None)
    parser.add_argument('-t', '--title', help='Title for the plot',
                        default='Interference from DC-MSS-IMT satellites to FS(2655MHz)')
    parser.add_argument('-x', '--marker', type=float, help='X value to mark on the plot', default=-6.0)

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return

    # 读取数据
    data = read_data_from_file(args.input_file)

    if len(data) == 0:
        print("Error: No valid data found in the input file.")
        return

    # 绘制CDF
    plot_cdf(data, args.output, args.title)


if __name__ == "__main__":
    main()