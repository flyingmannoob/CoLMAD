import os
import numpy as np

root_path = "./result"  

# 遍历指定目录下所有以 'e' 开头的子文件夹
e_folders = [name for name in os.listdir(root_path)
             if os.path.isdir(os.path.join(root_path, name)) and name.startswith('s')]

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

all_preds = []
all_trues = []

# 输出结果
for folder in e_folders:
    print(folder)
    anomaly_file = os.path.join(root_path,folder, "anomalies.txt")
    if os.path.exists(anomaly_file):
        with open(anomaly_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")  # 或者用 split() 默认空白分隔
                if len(parts) == 2:
                    pred_label = int(parts[0])
                    true_label = int(parts[1])
                    all_preds.append(pred_label)
                    all_trues.append(true_label)

precision, recall, f1 = range_f1(all_trues, all_preds)

print("Adjust result: ")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")