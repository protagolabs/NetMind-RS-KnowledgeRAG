""" 
@file_name: unit_test.py
@author: Yujing Wang, Bin Liang
@date: 2025-08-11
@description: 
    RAG智能代理单元测试模块
    
    本模块提供了RAG智能代理系统的基本测试用例，用于验证系统的
    核心功能是否正常工作。包括端到端的查询处理测试。
    
    测试内容：
    1. RAG代理初始化测试
    2. 查询处理流程测试
    3. 结果输出验证
    
    使用方法：
        python unit_test.py
    
    测试查询示例：
        - "Please sort out all the professional terms in these papers for me"
        - 这是一个信息汇总型查询，用于测试系统的术语提取和整理能力
"""

import json
from tqdm.auto import tqdm
from typing import List
from knowledge_rag.agentic_generation.rag_agent import RAGAgent


async def main():
    """主函数，使用异步上下文管理器确保资源正确释放"""
    
    # with open("data/questions.json", "r") as f:
    #     questions = json.load(f)
    query = "For the Patriot Exchange policy's Accidental Death & Dismemberment benefit, what is the maximum principal sum payable, and name two specific types of activities or conditions explicitly listed as exclusions for accidental death or dismemberment coverage? "
    print(query)
    async with RAGAgent() as rag_agent:
        result = await rag_agent.rag_agent(query, dataset_type="baoxian") 
    print(f"Chunk ids: {result['chunk_ids']}")
    print(f"Cost breakdown: {result['cost_breakdown']}")
    print(f"Performance metrics: {result['performance_metrics']}")
    print(f"Metadata: {result['metadata']}")
    # all_results = []
    # async with RAGAgent() as rag_agent:
    #     for question in tqdm(questions):
    #         query = question['question']
    #         result = await rag_agent.rag_agent(query) 
    #         question['xyz_answer'] = result['answer']
    #         question['chunks'] = result['chunk_ids']
    #         question['cost_breakdown'] = result['cost_breakdown']
    #         question['performance_metrics'] = result['performance_metrics']
    #         question['metadata'] = result['metadata']
    #         all_results.append(question)
            
    #         with open("data/results_20250815.json", "w") as f:
    #             json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    