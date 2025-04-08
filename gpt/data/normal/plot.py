import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import mplcursors  # 用于鼠标悬停显示坐标信息

file_name = 'case50'

# 读取 CSV 文件
df = pd.read_csv(f'./{file_name}.csv')


# 确保 AtTime 列是 datetime 类型
df['AtTime'] = pd.to_datetime(df['AtTime'])

# 设置 AtTime 列为索引
df.set_index('AtTime', inplace=True)

# 进行季节性分解，模型类型为 'additive' 或 'multiplicative'
decomposition = seasonal_decompose(df['UnhealthyCount'], model='additive', period=24)  # period=24 假设是每日24个数据点

# 获取趋势成分（trend）
df['Trend'] = decomposition.trend

# 选择两个时间点，在这两个时间点画垂直线
time_point1 = pd.to_datetime('2024-12-12T07:00:00Z')  # 替换为你选择的第一个时间点
time_point2 = pd.to_datetime('2024-12-13T05:00:00Z')  # 替换为你选择的第二个时间点

# 绘制趋势图
plt.figure(figsize=(12, 6))  # 建议将图表宽度加大以便显示所有刻度

plt.plot(df.index, df['Trend'], color='blue', label="Trend")  # 绘制趋势曲线

# 在两个时间点上画垂直线
plt.axvline(x=time_point1, color='red', linestyle='--', label='Start time')
plt.axvline(x=time_point2, color='red', linestyle='--', label='End time')

# 设置图例
plt.legend()

# **设置每小时一个刻度** 
hour_locator = mdates.HourLocator(interval=1)  # 每1小时一个主刻度
hour_formatter = mdates.DateFormatter('%m-%d %H:%M')  # 显示格式：月-日 时:分
ax = plt.gca()
ax.xaxis.set_major_locator(hour_locator)
ax.xaxis.set_major_formatter(hour_formatter)

plt.xticks(rotation=45, fontsize=8)  # 旋转刻度标签并设置字体大小

# **添加鼠标悬停功能**
cursor = mplcursors.cursor(hover=True)

# 自定义鼠标悬停时的提示信息
@cursor.connect("add")
def on_hover(sel):
    sel.annotation.set_text(
        f"Time: {mdates.num2date(sel.target[0]).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Trend: {sel.target[1]:.2f}"
    )

# 保存图形为 PNG 文件
plt.savefig(f'./trend_plot_{file_name}.png', bbox_inches='tight', dpi=300)

# 显示图形
plt.show()

# 如果你想保存归一化后的趋势数据，可以将其保存到新的 CSV 文件
df[['Trend']].to_csv(f'./trend_data_{file_name}.csv')
