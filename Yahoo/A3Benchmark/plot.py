import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 确保 plot 目录存在
os.makedirs("plot", exist_ok=True)

# 获取当前文件夹下所有 CSV 文件
csv_files = glob.glob("*.csv")

for file in csv_files:
    # 读取 CSV 文件
    df = pd.read_csv(file)
    
    # 确保所需的列存在，并计算 'Value' 列
    required_columns = ['value', 'trend', 'noise', 'seasonality1', 'seasonality2', 'seasonality3']
    if all(col in df.columns for col in required_columns):
        # 计算 'Value' 列
        df['Value'] = df['value'] - df['trend'] - df['noise'] - df['seasonality1'] - df['seasonality2'] - df['seasonality3']
    else:
        print(f"File {file} is missing one or more required columns.")
        continue

    # 绘制折线图
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamps"], df["Value"], color="blue", label="Value")

    # 标注 anomaly 点（红色）
    anomalies = df[df["anomaly"] == 1]
    plt.scatter(anomalies["timestamps"], anomalies["Value"], color="red", label="Anomaly", marker='o')

    # 标注正常点（绿色）
    normal = df[df["anomaly"] == 0]
    plt.scatter(normal["timestamps"], normal["Value"], color="green", label="Normal", marker='o')

    plt.xlabel("timestamps")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"Time Series with Anomalies: {file}")
    
    # 旋转 x 轴标签以防止重叠
    plt.xticks(rotation=45)

    # 保存图像
    plot_filename = os.path.join("plot", f"{os.path.splitext(file)[0]}.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()  # 关闭当前图像，避免内存占用

    print(f"Saved plot: {plot_filename}")
