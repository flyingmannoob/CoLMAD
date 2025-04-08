import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter


import mplcursors  # 用于鼠标悬停显示坐标信息

file_name = 'synthetic_29'

# 读取 CSV 文件
df = pd.read_csv(f'./{file_name}.csv')

# 重新编号 timestamp 列
df["timestamp"] = range(1, len(df) + 1)

# 归一化到 [0, 100]
min_val = df['value'].min()
max_val = df['value'].max()
df['value'] = (df['value'] - min_val) / (max_val - min_val) * 100

df['timestamp'] = df['timestamp'].astype(int)

# 进行季节性分解，模型类型为 'additive' 或 'multiplicative'
decomposition = seasonal_decompose(df['value'], model='additive', period=234)  # period=24 假设是每日24个数据点

# 获取趋势成分（trend）
df['stl'] =  decomposition.seasonal + decomposition.trend
# df['Smoothed'] = df['value'].rolling(window=10, center=True).mean()
# df['Smoothed'].fillna(method='bfill') 
df['Smoothed'] = savgol_filter(df['value'], window_length=10, polyorder=3)

# 绘制趋势图
plt.figure(figsize=(12, 6))  # 建议将图表宽度加大以便显示所有刻度

# plt.plot(df.index, df['stl'], color='blue', label="STL")  # 绘制趋势曲线
plt.plot(df.index, df['Smoothed'], color='red', label="Smoothed")  # 绘制趋势曲线
plt.plot(df.index, df['value'], color='green', linestyle='-', alpha=0.5, label="value")  # 添加原始数据折线

# 设置图例
plt.legend()

ax = plt.gca()

plt.xticks(rotation=45, fontsize=8)  # 旋转刻度标签并设置字体大小

# **添加鼠标悬停功能**
cursor = mplcursors.cursor(hover=True)

# # 保存图形为 PNG 文件
# plt.savefig(f'./trend_plot_{file_name}.png', bbox_inches='tight', dpi=300)

# 显示图形
plt.show()

# 如果你想保存归一化后的趋势数据，可以将其保存到新的 CSV 文件
df[['stl']].to_csv(f'./trend_data_{file_name}.csv')
