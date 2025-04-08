from openai import AzureOpenAI
import sys
import json
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from .cloudgpt_aoai import get_chat_completion

class Agent:
    def __init__(self) -> None:
        self.engine="o1-20241217"

    @staticmethod
    def to_json_str(dict_obj):
        return json.dumps(dict_obj, indent=4, separators=(',', ': '))
    
    def generate_response(self, message):
        chat_message = [{"role": "user", "content": message}]
        response = get_chat_completion(
            engine = self.engine,
            messages = chat_message,
        )
        
        return response.choices[0].message
    
class Code_generator(Agent):
    def __init__(self) -> None:
        super().__init__()
    def generate(self, data_descriptions, method_descriptions, detection_result, window_size, error_mechanism):
            
        prompt = f'''
            You are given a description of a time series data, an anomaly detection method, its detection results, the sampling window size, and a high-level explanation of the method's error mechanism (e.g., conditions under which it produces false positives or false negatives).
            [Task]: Your goal is to determine which points in the time series are anomalies based on the following rules:
            [Inputs]:
                [description of a time series data]: {repr(data_descriptions)} 
                [description of an anomaly detection method]: {repr(method_descriptions)}
                [method's detection results]: {repr(detection_result)}
                [method's sampling window size]: {repr(window_size)}
                [high-level explanation of the method's error mechanism]: {repr(error_mechanism)}
            [Rules for Identifying Anomalies]:
             1. Definition of Normal Pattern:
                - **normal pattern 1** is the value that **most points** in the time series **maintain** or **fluctuate around** within a small range. Minor fluctuations around a fixed value (e.g., `4 - 6` when the majority value is `5` are considered normal.) Example of normal pattern: `5, 5, 5, 4, 5.1, 5, 5, 6, 5.2, 5, 5` (small fluctuations, but centered around 5).
                - **normal pattern 2** is also that regular periodic fluctuations (every period contains a linear up-down or down-up wave). Note: (1) Maybe with a gradually small rising trend in every period's max vaule. (2) Small fluctuations around the maximum value point are also considered normal. (3) If the maximum (or minimum) values of each period do not differ significantly, the series can also be considered as exhibiting periodic fluctuations. 
                - The regular periodic time series does not contain an integer number of full periods; that is, it may 'start' from a partial period, go through several complete periods, and 'end' with another partial period. You need to get the normal pattern from the complete periods among them, and don not think the partial periods for 'start' and 'end' as anomaly.
                - 'Piecewise stable plateaus with short transitions between them' is not **normal pattern**.  
             2. Anomalous points deviate significantly from normal points. A point is considered anomalous if it either: 
                - (1) A group of consecutive points deviates from the normal pattern, such as:
                    - Prolonged abnormal fluctuations(linear transitions) (e.g., 5 → 12 → 18 → 24 → 16 → 8 → 5).
                - (2) Isolated anomaly
                    - A single point suddenly jumps far from the normal range (e.g., 5, 5, 5, *50*, 5, 5, 5).
                - (3) Period deviation anomaly
                    - A single point or group of consecutive points jump far away the regular periodic pattern. Such as a spike in the up(down) wave.
             3. Anomalies typically exhibit continuity: they start as a small deviation, progressively become more extreme, reach a peak deviation, and then gradually return to normal values. When you identify an anomaly, you must capture the entire anomalous segment, from the initial deviation → peak deviation → return to normal.
             4. The provided anomaly detection method is not always correct.
             5. To simplify the description, let's describe the data points with some fluctuations as a linear line. So, don't just think of linear changes as normal patterns.
            [Instructions]: Follow these steps to make your final anomaly detection:
             1. Understand the natural language description of the time series and extract the normal data pattern from it. Strictly adhere to my definition of normal pattern.
             2. Analyze the given anomaly detection method, paying close attention to its error mechanism—the conditions under which it produces false positives or false negatives.
             3. Based on both the provided error mechanism and the actual time series data, decide the results of the method is right or wrong. 
             4. If wrong, rethink the final anomaly detection of the whole time series data without the results of the method, just use the differences between the normal and anomaly pattern. 
            [Output Format]: Your response should be a valid JSON object formatted as a single line, with no extra spaces or line breaks. The JSON object should contain a single key `matches`. The value of `matches` should be a dictionary containing:
            {{
                "normal pattern": "<pattern>", // A summary of the normal pattern
                "decision": "right" | "wrong",  // "right" if you think the method's result is correct, otherwise "wrong"
                "reason_for_decision": "<explanation>",  // An explanation of why you made this decision
                "anomalies": [
                    [start_index, end_index],  // If the anomaly is a range, provide the start and end index
                    single_index,  // If the anomaly is an isolated point, provide just the index
                    ...
                ], // all points you consider anomalous
                "reason_for_anomaly": "<explanation>"  // An explanation of why these points are considered anomalous
            }}
            Please ensure that the response is a single line and does not include extra characters like newlines or indentation. The response should be properly formatted JSON.
        '''

        response = self.generate_response(prompt)

        try:
            
            response = response.content
            response = json.loads(response)
            response = response['matches']
            print(response)
            
        except Exception as e:
            print(f"Method Selector converts {response} to JSON error!")
            print(e)
            response = None
        return response
    
if __name__ == "__main__":

    GPT_CONFIG = ''
    generator = Code_generator()
    data_descriptions = "the time series consists of 1439 data points:from index 1 to 1359, the value remains constant at 5;from index 1359 to 1363, it changes linearly from 4 to 100 (slope: 19.09 per step);from index 1363 to 1364, it changes linearly from 100 to 4 (slope: -47.79 per step);from index 1364 to 1426, the value remains constant at 4.0;from index 1426 to 1427, it changes linearly from 2 to 19 (slope: 8.67 per step);from index 1427 to 1429, it changes linearly from 19 to 5 (slope: -4.65 per step);from index 1429 to 1434, it changes linearly from 5 to 89 (slope: 13.98 per step);from index 1434 to 1439, it changes linearly from 89 to 26 (slope: -10.63 per step)."
    method_descriptions = "The 3-sigma method identifies anomalies by assuming a normal distribution and classifying any data point beyond three standard deviations from the mean as an outlier."
    detection_result = "The 3-sigma method detected anomalies at range 1359 - 1439, other points are normal."
    window_size = "200"
    error_mechanism = "False negative scenario: If a sampling window contains many extremely high or low values, they can significantly raise or lower the mean and increase the standard deviation, making truly anomalous points appear less abnormal and thus not detected. False positive scenario: If the time series naturally exhibits high volatility but is not actually anomalous, it may still be incorrectly flagged as an anomaly for exceeding the kσ threshold."
    response_json = generator.generate(data_descriptions, method_descriptions, detection_result, window_size, error_mechanism)
    
    # 处理anomalies
    anomalies = response_json['anomalies']
    
    length = 1439
    result = [0] * length

    for item in anomalies:
        if isinstance(item, list):  # 如果是区间
            for i in range(item[0], item[1] + 1):
                if 0 <= i < length:
                    result[i] = 1
        else:  # 如果是单个值
            if 0 <= item < length:
                result[item] = 1
    
    # 指定要保存的文件名
    output_filename = "output.json"
    # 将 JSON 数据写入文件
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=4, ensure_ascii=False)

    print(f"JSON data successfully saved to {output_filename}")

