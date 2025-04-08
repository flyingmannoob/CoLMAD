import pandas as pd
import numpy as np
from scipy import stats
import os

class SigmaDetector:
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect(self, df):
        # 确保 'value' 列是数值类型
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # 使用前800个点计算均值和标准差
        mean_initial = df['value'].iloc[0:500].mean()
        std_initial = df['value'].iloc[0:500].std()

        # 使用初始的均值和标准差计算所有点的Z-score
        df['ZScore'] = (df['value'] - mean_initial) / std_initial

        # 打印Z-score用于调试（可选）
        print(df['ZScore'].iloc[0:2])

        # 标记异常值
        df['label'] = np.where(df['ZScore'].abs() > self.threshold, 1, 0)

        # 修改列名，'value' 改为 'vaule'
        df.rename(columns={'value': 'vaule'}, inplace=True)

        # 返回包含 index, vaule, label 和 ZScore 的 DataFrame
        return df[['Index', 'vaule', 'label', 'ZScore']]

# 示例用法
if __name__ == "__main__":
    
    # 创建父文件夹的父文件夹中的 result 文件夹（如果不存在）
    result_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    detection_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result', 'anomaly_detection_result')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(detection_folder):
        os.makedirs(detection_folder)

    # 这里使用一个示例数据框
    df = pd.read_csv(r'C:\Users\v-xiaoyufeng\Desktop\gpt_detection\process_data\real_2_process_data.csv')
    
    # 创建检测器实例
    detector = SigmaDetector(threshold=3)
    
    # 调用检测方法
    anomalies = detector.detect(df)
    
    # 保存结果到父文件夹下的result文件夹
    result_file = os.path.join(detection_folder, 'k-sigma_result.csv')
    anomalies.to_csv(result_file, index=False)
    
    # 打印结果
    print(anomalies.head())
