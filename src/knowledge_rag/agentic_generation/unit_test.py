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

from typing import List
from knowledge_rag.agentic_generation.rag_agent import RAGAgent


if __name__ == "__main__":
    import asyncio
    rag_agent = RAGAgent()
    query = "Please sort out all the professional terms in these papers for me"
    result = asyncio.run(rag_agent.rag_agent(query))
    print(result)