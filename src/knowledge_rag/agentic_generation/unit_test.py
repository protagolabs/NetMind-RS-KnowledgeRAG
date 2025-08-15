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


async def main():
    """主函数，使用异步上下文管理器确保资源正确释放"""
    async with RAGAgent() as rag_agent:
        query = "What are some prominent pre-training approaches in natural language processing mentioned in the document, and how do they differ in their objective or methodology?"
        result = await rag_agent.rag_agent(query)
        print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    