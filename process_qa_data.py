#!/usr/bin/env python3
"""
处理QA结果数据的脚本

功能：
1. 从原始JSON文件中提取 'single_doc_multi_chunk' 和 'multi_doc_multi_chunk' 的数据
2. 保持原有的大类结构
3. 在每个大类内部按 'question_type' 分类
4. 每个类型选择2个数据
5. 生成新的JSON文件
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any

def process_qa_data(input_file: str, output_file: str, max_per_type: int = 2):
    """
    处理QA数据
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        max_per_type: 每个问题类型最多选择的数据数量
    """
    
    # 读取原始数据
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取指定key的数据
    target_keys = ['single_doc_multi_chunk', 'multi_doc_multi_chunk']
    result = {}
    
    for key in target_keys:
        if key in data:
            print(f"\n处理 {key}:")
            key_data = data[key]
            
            # 按question_type分类
            categorized_data = defaultdict(list)
            
            for item in key_data:
                question_type = item.get('question_type', 'Unknown')
                categorized_data[question_type].append(item)
            
            print(f"按问题类型分类结果:")
            for qtype, items in categorized_data.items():
                print(f"  {qtype}: {len(items)} 条")
            
            # 每个类型选择指定数量的数据
            key_result = {}
            
            for question_type, items in categorized_data.items():
                # 选择前max_per_type个数据
                selected_items = items[:max_per_type]
                key_result[question_type] = selected_items
                print(f"为 {question_type} 选择了 {len(selected_items)} 条数据")
            
            result[key] = key_result
            
        else:
            print(f"警告: 未找到key '{key}'")
            result[key] = {}
    
    # 保存结果
    print(f"\n正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("处理完成!")
    
    # 打印统计信息
    print("\n最终统计:")
    for key, key_data in result.items():
        print(f"\n{key}:")
        for qtype, items in key_data.items():
            print(f"  {qtype}: {len(items)} 条")
    
    return result

def main():
    """主函数"""
    # 文件路径
    input_file = "src/knowledge_rag/test_QA_results_en.json"
    output_file = "src/knowledge_rag/processed_qa_data.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 处理数据
    try:
        result = process_qa_data(input_file, output_file, max_per_type=2)
        print(f"\n处理成功! 结果已保存到: {output_file}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 