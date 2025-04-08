import pandas as pd

def describe_sigma_results(file_name):
    # 读取 CSV 文件
    df = pd.read_csv(file_name)
    df['Index'] = df['Index'].astype(int)
    
    # 确保列存在
    if 'Index' not in df.columns or 'label' not in df.columns:
        return "Invalid file format. Columns 'index' and 'label' are required."
    
    # 获取异常点
    anomalies = df[df['label'] == 1]['Index'].tolist()
    
    # 处理异常情况
    if len(anomalies) == 0:
        return "The 3-sigma method results show that all data points are normal."
    
    # 识别异常区间
    ranges = []
    start = anomalies[0]
    prev = start
    
    for i in range(1, len(anomalies)):
        if anomalies[i] == prev + 1:
            prev = anomalies[i]
        else:
            ranges.append((start, prev))
            start = anomalies[i]
            prev = start
    ranges.append((start, prev))
    print(ranges)
    
    # 生成描述
    descriptions = []
    for start, end in ranges:
        if start == end:
            descriptions.append(f"point {start}")
        else:
            descriptions.append(f"range {start}-{end}")
    
    return "The 3-sigma method detected anomalies at " + ", ".join(descriptions) + "," + 'other points are normal.'

if __name__ == "__main__":
    # 示例调用
    file_name = "../result/anomaly_detection_result/k-sigma_result.csv"  # 请替换为实际文件路径
    description = describe_sigma_results(file_name)
    print(description)
