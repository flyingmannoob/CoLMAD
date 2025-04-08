import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors

# 数据同化处理
def homogenize_data(points, threshold=3):
    homogenized_points = []
    prev_x, prev_y = points[0]
    homogenized_points.append((prev_x, prev_y))

    for x, y in points[1:]:
        if abs(y - prev_y) <= threshold:
            homogenized_points.append((x, prev_y))  # 同化
        else:
            homogenized_points.append((x, y))
            prev_y = y  # 更新前一个点的值

    return homogenized_points

# Douglas-Peucker 压缩算法
def douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points
    
    start, end = 0, len(points) - 1
    max_dist = 0
    idx_far = 0
    
    x1, y1 = points[start]
    x2, y2 = points[end]
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    for i in range(start + 1, end):
        x0, y0 = points[i]
        dist = abs(A * x0 + B * y0 + C) / np.sqrt(A * A + B * B)
        if dist > max_dist:
            max_dist = dist
            idx_far = i
    
    if max_dist > epsilon:
        left_pts = douglas_peucker(points[start:idx_far + 1], epsilon)
        right_pts = douglas_peucker(points[idx_far:end + 1], epsilon)
        return left_pts[:-1] + right_pts
    else:
        return [points[start], points[end]]

# 线性插值恢复数据
def restore_points(original_points, simplified_points):
    restored_points = []
    
    for i in range(len(simplified_points) - 1):
        x1, y1 = simplified_points[i]
        x2, y2 = simplified_points[i + 1]
        
        restored_points.append((x1, y1))
        segment_points = [pt for pt in original_points if x1 < pt[0] < x2]
        
        for x in sorted(segment_points, key=lambda p: p[0]):
            y = np.interp(x[0], [x1, x2], [y1, y2])
            restored_points.append((x[0], y))
    
    restored_points.append(simplified_points[-1])
    return restored_points

def simplify_low_slope_segments(vaule, points, threshold=2):
    """
    如果整个段的平均斜率绝对值小于 threshold，则用一条水平直线简化该段
    """
    if len(points) < 2:
        return points  # 只有一个点，无法简化
    
    # x_start, y_start = points[0]
    x_end, y_end = points[-1]
    
    # # 计算整个段的平均斜率
    # slope = (y_end - y_start) / len(points)

    slope = y_end - vaule

    # 低斜率情况，简化为水平线
    if abs(slope) < threshold:
        return [(x[0], vaule) for x in points]  # 所有点的 y 设为 y_start，x 保持不变
    
    return points  # 否则返回原始点集

def convert_to_natural_language(points):
    """
    将低斜率简化后的时间序列转换为自然语言描述格式。
    """
    description = ''
    
    x_start, y_start = points[0]
    x_end, y_end = points[-1]

    # 计算斜率
    slope = (y_end - y_start) / len(points)
    # print(slope)

    if round(slope, 2) == 0.00 :  # 为水平直线
        description += f"from index {round(x_start)} to {round(x_end)}, the value remains constant at {round(y_start)};"
    elif abs(round(slope, 2)) < 0.05: # 近似为水平直线
        description += f"from index {round(x_start)} to {round(x_end)}, the value remains constant at {(round(y_start)+ round(y_end))/2};"
    else:  # 斜率不为 0，是一条折线
        description += f"from index {round(x_start)} to {round(x_end)}, it changes linearly from {round(y_start)} to {round(y_end)} (slope: {slope:.2f} per step);"

    return description


if __name__ == "__main__":

    # For Yahoo A2
    file = '../Yahoo/A2Benchmark/synthetic_49'
    # file = '../pattern/case43'
    file_name = file + '.csv'

    # 读取数据
    df = pd.read_csv(
        file_name,
        parse_dates=['timestamp'],
        dtype={'value': float}
    )

    # df = df[134:1381]

    # 重新编号 timestamp 列
    df["timestamp"] = range(1, len(df) + 1)

    print(len(df))

    # 归一化到 [0, 100]
    min_val = df['value'].min()
    max_val = df['value'].max()
    df['value'] = (df['value'] - min_val) / (max_val - min_val) * 100

    df['timestamp'] = df['timestamp'].astype(int)
    
    points = list(zip(df['timestamp'], df['value'])) 

    # 应用数据同化
    points = homogenize_data(points, threshold=3)
        
    # 设置 Douglas-Peucker 压缩精度
    epsilon = 4
    simplified_pts = np.array(douglas_peucker(points, epsilon))
    
    # 获取简化后的拐点索引（即压缩后的点）
    simplified_indices = [pt[0] for pt in simplified_pts]  # 保存压缩后的点的索引
    
    # 恢复数据
    restored_pts = restore_points(points, simplified_pts)

    # **找到恢复后的数据中，每两个拐点之间的点**
    final_pts = []
    
    st_index = int(simplified_indices[0])
    st_vaule = restored_pts[st_index][1]
    vaule = st_vaule
    print(vaule)

    final_descriptions = ''
    final_descriptions += f'the time series consists of {len(restored_pts)} data points:'

    for i in range(len(simplified_indices) - 1):
        start_idx = simplified_indices[i]
        end_idx = simplified_indices[i + 1]
        
        # 找到恢复后的数据中位于 start_idx 和 end_idx 之间的所有点
        segment_points = [pt for pt in restored_pts if start_idx <= pt[0] <= end_idx]
        
        # 在这些点上进行低斜率简化
        simplified_segment = simplify_low_slope_segments(vaule, segment_points, threshold=1)

        description = convert_to_natural_language(segment_points)
        
        vaule = simplified_segment[-1][1]

        if i != 0:
            simplified_segment = simplified_segment[1:]
        
        # 存入最终结果
        final_pts.extend(simplified_segment)
        final_descriptions += description

    final_pts = np.array(final_pts)
    final_descriptions = final_descriptions[:-1] + '.'  # 去掉最后一个字符并替换为 '.'
    print(final_descriptions)
    
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['value'], 'o-', alpha=0.3, label='origin')
    # ax.plot(simplified_pts[:, 0], simplified_pts[:, 1], 'r-o', label='simple')
    # ax.plot([pt[0] for pt in restored_pts], [pt[1] for pt in restored_pts], 'g--', label='恢复数据')
    ax.plot(final_pts[:, 0], final_pts[:, 1], 'b-', label='process')
    
    plt.title(f'Douglas-Peucker (ε={epsilon})')
    plt.ylabel('value Value')
    plt.legend()
    plt.tight_layout()
    # 添加鼠标悬停功能
    mplcursors.cursor(hover=True)
    plt.show()
    
    # # For Yahoo A1
    # file = '../Yahoo/A1Benchmark/real_65'
    # file_name = file + '.csv'

    # # 读取数据
    # df = pd.read_csv(
    #     file_name,
    #     parse_dates=['timestamp'],
    #     dtype={'value': float}
    # )
    
    # df.dropna(subset=['timestamp'], inplace=True)
    # df['value'].fillna(0, inplace=True)

    # # 归一化到 [0, 100]
    # min_val = df['value'].min()
    # max_val = df['value'].max()
    # df['value'] = (df['value'] - min_val) / (max_val - min_val) * 100

    # df['timestamp'] = df['timestamp'].astype(int)
    
    # points = list(zip(df['timestamp'], df['value'])) 

    # # 应用数据同化
    # points = homogenize_data(points, threshold=3)
        
    # # 设置 Douglas-Peucker 压缩精度
    # epsilon = 7
    # simplified_pts = np.array(douglas_peucker(points, epsilon))
    
    # # 获取简化后的拐点索引（即压缩后的点）
    # simplified_indices = [pt[0] for pt in simplified_pts]  # 保存压缩后的点的索引
    
    # # 恢复数据
    # restored_pts = restore_points(points, simplified_pts)

    # # **找到恢复后的数据中，每两个拐点之间的点**
    # final_pts = []
    
    # st_index = int(simplified_indices[0])
    # st_vaule = restored_pts[st_index][1]
    # vaule = st_vaule
    # print(vaule)

    # final_descriptions = ''
    # final_descriptions += f'the time series consists of {len(restored_pts)} data points:'

    # for i in range(len(simplified_indices) - 1):
    #     start_idx = simplified_indices[i]
    #     end_idx = simplified_indices[i + 1]
        
    #     # 找到恢复后的数据中位于 start_idx 和 end_idx 之间的所有点
    #     segment_points = [pt for pt in restored_pts if start_idx <= pt[0] <= end_idx]
        
    #     # 在这些点上进行低斜率简化
    #     simplified_segment = simplify_low_slope_segments(vaule, segment_points, threshold=1)

    #     description = convert_to_natural_language(segment_points)
        
    #     vaule = simplified_segment[-1][1]

    #     if i != 0:
    #         simplified_segment = simplified_segment[1:]
        
    #     # 存入最终结果
    #     final_pts.extend(simplified_segment)
    #     final_descriptions += description

    # final_pts = np.array(final_pts)
    # final_descriptions = final_descriptions[:-1] + '.'  # 去掉最后一个字符并替换为 '.'
    # print(final_descriptions)
    
    
    # # 绘图
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(df.index, df['value'], 'o-', alpha=0.3, label='origin')
    # # ax.plot(simplified_pts[:, 0], simplified_pts[:, 1], 'r-o', label='simple')
    # # ax.plot([pt[0] for pt in restored_pts], [pt[1] for pt in restored_pts], 'g--', label='恢复数据')
    # ax.plot(final_pts[:, 0], final_pts[:, 1], 'b-', label='process')
    
    # plt.title(f'Douglas-Peucker (ε={epsilon})')
    # plt.ylabel('value Value')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # # 保存最终简化后的数据到 CSV
    # # final_df = pd.DataFrame(final_pts, columns=['Index', 'value'])
    # # final_df.to_csv(f'{file}_process_data.csv', index=False)

    # # print(f"最终简化数据已保存至 {file}_process_data.csv")

    
    # file_name = 'real_1.csv'

    # # 读取数据
    # df = pd.read_csv(
    #     file_name,
    #     parse_dates=['timestamp'],
    #     date_parser=lambda x: pd.to_datetime(x, utc=True),
    #     dtype={'value': float}
    # )
    
    # df.dropna(subset=['timestamp'], inplace=True)
    # df['value'].fillna(0, inplace=True)
    
    # # 使用索引替代时间戳
    # points = list(zip(df.index, df['value']))  # 使用DataFrame的索引作为x值
    
    # # 设置 Douglas-Peucker 压缩精度
    # epsilon = 10
    # simplified_pts = np.array(douglas_peucker(points, epsilon))
    
    # # 获取简化后的拐点索引（即压缩后的点）
    # simplified_indices = [pt[0] for pt in simplified_pts]  # 保存压缩后的点的索引
    
    # # 恢复数据
    # restored_pts = restore_points(points, simplified_pts)

    # # **找到恢复后的数据中，每两个拐点之间的点**
    # final_pts = []
    
    # st_index = int(simplified_indices[0])
    # st_vaule = restored_pts[st_index][1]
    # vaule = st_vaule
    # print(vaule)

    # final_descriptions = ''
    # final_descriptions += f'the time series consists of {len(restored_pts)} data points:'

    # for i in range(len(simplified_indices) - 1):
    #     start_idx = simplified_indices[i]
    #     end_idx = simplified_indices[i + 1]
        
    #     # 找到恢复后的数据中位于 start_idx 和 end_idx 之间的所有点
    #     segment_points = [pt for pt in restored_pts if start_idx <= pt[0] <= end_idx]
        
    #     # 在这些点上进行低斜率简化
    #     simplified_segment = simplify_low_slope_segments(vaule, segment_points, threshold=1)

    #     description = convert_to_natural_language(segment_points)
        
    #     vaule = simplified_segment[-1][1]
        
    #     # 存入最终结果
    #     final_pts.extend(simplified_segment)
    #     final_descriptions += description

    # final_pts = np.array(final_pts)
    # final_descriptions = final_descriptions[:-1] + '.'  # 去掉最后一个字符并替换为 '.'
    # print(final_descriptions)
    
    
    # # 绘图
    # fig, ax = plt.subplots(figsize=(12, 6))
    # # ax.plot(df.index, df['value'], 'o-', alpha=0.3, label='原始数据')
    # # ax.plot(simplified_pts[:, 0], simplified_pts[:, 1], 'r-o', label='简化轨迹')
    # # ax.plot([pt[0] for pt in restored_pts], [pt[1] for pt in restored_pts], 'g--', label='恢复数据')
    # ax.plot(final_pts[:, 0], final_pts[:, 1], 'b-', label='低斜率简化')
    
    # plt.title(f'Douglas-Peucker 压缩 & 线性插值恢复 & 低斜率简化 (ε={epsilon})')
    # plt.ylabel('value Value')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # 保存最终简化后的数据到 CSV
    # final_df = pd.DataFrame(final_pts, columns=['Index', 'value'])
    # final_df.to_csv(f'{file_name}_process_data.csv', index=False)

    # print(f"最终简化数据已保存至 {file_name}_process_data.csv")
