
import requests
import json
import os
import pandas as pd
import numpy as np
from langchain.text_splitter import CharacterTextSplitter

# 替换为你的API凭据
access_token="111"
key="111"
secret_key="111"

# 定义疾病诊断标准变量
diagnosis_criteria = "肥胖的诊断标准为BMI大于等于28"

# 发送患者数据和诊断标准给API并获取诊断结果
def get_diagnosis(patient_data, diagnosis_criteria):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "inputs": [
            {
                "text": patient_data,
                "question": f"根据以下诊断标准判断这个病人是否患有该疾病：{diagnosis_criteria}"
            }
        ],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json().get("result")
    # 如果响应结果包含“是”则返回“1”，否则返回“0”
   

# 处理Excel文件，判断每个病人是否患有该疾病
def process_excel(file_path, diagnosis_criteria, output_path):
    df = pd.read_excel(file_path)
    diagnosis_results = []
    
    for index, row in df.iterrows():
        patient_data = ' '.join(row.astype(str).tolist())
        result = get_diagnosis(patient_data, diagnosis_criteria)
        diagnosis_results.append(result)


    df['Diagnosis'] = diagnosis_results
    df.to_excel(output_path, index=False)
    print("Results have been written to the output Excel file.")

# 主函数
def main():
    input_excel_path = "C:\\Users\\40467\\Desktop\\py_task\\input.xlsx"
    output_excel_path = "C:\\Users\\40467\\Desktop\\py_task\\output.xlsx"
    
    process_excel(input_excel_path, diagnosis_criteria, output_excel_path)

# 判断是否为主程序入口
if __name__ == '__main__':
    main()
