import pandas as pd
from gpt.gpt_detection import Code_generator
from sklearn.metrics import precision_score, recall_score, f1_score
from process_data.test import homogenize_data
from process_data.test import douglas_peucker
from process_data.test import restore_points
from process_data.test import simplify_low_slope_segments
from process_data.test import convert_to_natural_language
from anomaly_detection.sigma_detector import SigmaDetector
from anomaly_detection.util import describe_sigma_results
import json
import os
import numpy as np

def parse_anomalies(anomalies, result):
    for item in anomalies:
        if isinstance(item, list):  # 处理区间
            print(item)
            for i in range(item[0], item[1]+1):
                result[i-1] = 1
        elif isinstance(item, int):  # 处理单个点
            if 0 <= item < len(result):
                result[item-1] = 1

if __name__ == "__main__":

    # For Yahoo
    groudtruth_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Yahoo', 'A2Benchmark')
    groudtruth = 'synthetic_49'
    # groudtruth_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pattern')
    # groudtruth = 'case43'
    groudtruth_file = os.path.join(groudtruth_folder, groudtruth + '.csv')
    groundtruth_df = pd.read_csv(groudtruth_file)
    # groundtruth_df = groundtruth_df[177:1342] # 22case

    # 保存结果
    result_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', groudtruth)
    process_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', groudtruth, 'process_data')
    detection_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', groudtruth, 'anomaly_detection_result')
    gpt_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', groudtruth, 'gpt_result')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(process_folder):
        os.makedirs(process_folder)
    if not os.path.exists(detection_folder):
        os.makedirs(detection_folder)
    if not os.path.exists(gpt_folder):
        os.makedirs(gpt_folder)

    result_file = os.path.join(detection_folder, 'k-sigma_result.csv')

    # step1: 数据处理
    # 读取数据
    df = groundtruth_df
    
    # 重新编号 timestamp 列
    df["timestamp"] = range(1, len(df) + 1)

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

    # 保存最终简化后的数据到 CSV
    final_df = pd.DataFrame(final_pts, columns=['Index', 'value'])
    final_df.to_csv(f'{process_folder}/process_data.csv', index=False)

    print(f"最终简化数据已保存至 process_data.csv")

    # step2: 异常检测
    # 创建检测器实例
    detector = SigmaDetector(threshold=3)
    
    # 调用检测方法
    anomalies = detector.detect(final_df)
    
    # 保存结果到result文件夹
    anomalies.to_csv(result_file, index=False)

    detection_description = describe_sigma_results(result_file)
    print(detection_description)

    # step3: GPT异常检测
    GPT_CONFIG = ''
    generator = Code_generator()
    data_descriptions = final_descriptions
    # data_descriptions = "the time series consists of 1439 data points:from index 1 to 1359, the value remains constant at 5;from index 1359 to 1363, it changes linearly from 4 to 100 (slope: 19.09 per step);from index 1363 to 1364, it changes linearly from 100 to 4 (slope: -47.79 per step);from index 1364 to 1426, the value remains constant at 4.0;from index 1426 to 1427, it changes linearly from 2 to 19 (slope: 8.67 per step);from index 1427 to 1429, it changes linearly from 19 to 5 (slope: -4.65 per step);from index 1429 to 1434, it changes linearly from 5 to 89 (slope: 13.98 per step);from index 1434 to 1439, it changes linearly from 89 to 26 (slope: -10.63 per step)."
    method_descriptions = "The 3-sigma method identifies anomalies by assuming a normal distribution and classifying any data point beyond three standard deviations from the mean as an outlier."
    detection_result = detection_description
    # detection_result = "The 3-sigma method detected anomalies at range 1359-1363, range 1363-1364, range 1364-1426, range 1426-1427, range 1427-1429, range 1429-1434, range 1434-1439,other points are normal."
    window_size = "200"
    error_mechanism = "False negative scenario: If a sampling window contains many extremely high or low values, they can significantly raise or lower the mean and increase the standard deviation, making truly anomalous points appear less abnormal and thus not detected. False positive scenario: If the time series naturally exhibits high volatility but is not actually anomalous, it may still be incorrectly flagged as an anomaly for exceeding the kσ threshold."
    response_json = generator.generate(data_descriptions, method_descriptions, detection_result, window_size, error_mechanism)
    
    # 指定要保存的文件名
    output_filename = f"{gpt_folder}/gpt_output.json"
    # 将 JSON 数据写入文件
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=4, ensure_ascii=False)

    # step4: 评估结果
    # 读取 JSON 文件
    with open(f"{gpt_folder}/gpt_output.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # 处理anomalies
    anomalies = data['anomalies']
    
    length = len(groundtruth_df)
    result = [0] * length

    parse_anomalies(data["anomalies"], result)

    true_labels = groundtruth_df['is_anomaly']

    sigma_result_df = pd.read_csv(result_file)
    sigma_result = sigma_result_df['label'].tolist()
    # 计算 Precision, Recall, F1-score
    precision = precision_score(true_labels, sigma_result)
    recall = recall_score(true_labels, sigma_result)
    f1 = f1_score(true_labels, sigma_result)
    print("k-sigma result: ")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


    # 计算 Precision, Recall, F1-score
    precision = precision_score(true_labels, result)
    recall = recall_score(true_labels, result)
    f1 = f1_score(true_labels, result)
    print("GPT result: ")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # 以 tab 分隔 result 和 true_labels 并保存到 anomalies.txt
    with open(f"{result_folder}/anomalies.txt", "w") as f:
        for r, t in zip(result, true_labels):
            f.write(f"{r}\t{t}\n")

    def get_anomaly_ranges(labels):
        ranges = []
        in_range = False
        for i, val in enumerate(labels):
            if val == 1 and not in_range:
                start = i
                in_range = True
            elif val == 0 and in_range:
                end = i
                ranges.append((start, end))
                in_range = False
        if in_range:
            ranges.append((start, len(labels)))
        return ranges

    def range_f1(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        true_ranges = get_anomaly_ranges(y_true)
        pred_ranges = get_anomaly_ranges(y_pred)

        TP = 0
        matched_true = set()
        
        for pred_range in pred_ranges:
            for i, true_range in enumerate(true_ranges):
                # 判断预测区间和真实区间是否有交集
                if pred_range[1] > true_range[0] and pred_range[0] < true_range[1]:
                    TP += 1
                    matched_true.add(i)
                    break

        FP = len(pred_ranges) - TP
        FN = len(true_ranges) - len(matched_true)
        
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return precision, recall, f1
    precision, recall, f1 = range_f1(true_labels, result)

    print("Adjust result: ")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")