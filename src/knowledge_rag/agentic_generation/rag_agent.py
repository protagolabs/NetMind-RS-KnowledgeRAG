""" 
@file_name: rag_agent.py
@author: Yujing Wang, Bin Liang
@date: 2025-08-11
@description: 
    RAG智能代理主模块
    
    本模块实现了基于智能代理的检索增强生成(RAG)系统，通过意图识别、查询改写、
    知识检索和答案生成四个核心步骤，为用户问题提供高质量的回答。
    
    主要功能：
    1. 意图识别：判断用户问题是事实型还是信息汇总型
    2. 查询改写：基于意图和文档摘要生成多个改写查询
    3. 知识检索：从知识库中检索相关文档和文本块
    4. 答案生成：基于检索结果生成结构化答案
    
    系统架构：
    - IntentRecognitionAgent：意图识别代理
    - QueryRewriteAgent：查询改写代理  
    - GenerationAgent：答案生成代理
    - retrieval_agent：知识检索客户端
"""

from loguru import logger
from typing import List, Tuple, Dict, Any
from time import time
import json

import asyncio
from knowledge_rag.agentic_generation.rag_agents import IntentRecognitionAgent, QueryRewriteAgent, GenerationAgent
from knowledge_rag.knowledge_retrieval.db_retriever_client import create_client
from knowledge_rag.main_process import make_decision_of_chunk_retrieval

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init(
    app_name="knowledge_rag",
    # api_key="tl_1e636be3e4dd41c2b4d3e9dad4b6ae6f"
    api_key="tl_46cc5aaa8f5649d89e530bcc6e2ac37b"
)

class RAGAgent:
    """RAG智能代理核心类。
    
    本类整合了意图识别、查询改写、知识检索和答案生成四个核心组件，
    提供端到端的智能问答服务。通过多步骤的处理流程，确保为用户
    提供准确、全面的回答。
    
    Attributes:
        intent_recognition_agent: 意图识别代理，判断问题类型
        query_rewrite_agent: 查询改写代理，生成多个检索查询
        generation_agent: 答案生成代理，基于检索结果生成答案
        retrieval_agent: 检索客户端，从知识库检索相关内容
    
    处理流程：
        1. 意图识别：分析用户问题的意图类型
        2. 文档检索：根据数据集类型检索相关文档
        3. 查询改写：基于意图和文档摘要生成多个改写查询
        4. 内容检索：使用改写查询检索相关文本块
        5. 答案生成：基于检索内容生成最终答案
    """
    def __init__(self):
        self.intent_recognition_agent = IntentRecognitionAgent()
        self.query_rewrite_agent = QueryRewriteAgent()
        self.generation_agent = GenerationAgent()
        
        self.retrieval_agent = create_client(
            # base_url="http://localhost:8955",
            base_url="http://71.178.110.3:8955",
            timeout=30.0,
            max_retries=3,
            retry_delay=1.0
        )
    
    async def close(self):
        """关闭所有客户端连接"""
        try:
            # 关闭OpenAI客户端
            if hasattr(self.intent_recognition_agent, 'openai_client'):
                await self.intent_recognition_agent.openai_client.close()
            if hasattr(self.query_rewrite_agent, 'openai_client'):
                await self.query_rewrite_agent.openai_client.close()
            if hasattr(self.generation_agent, 'openai_client'):
                await self.generation_agent.openai_client.close()
            
            # 关闭检索客户端
            if hasattr(self.retrieval_agent, 'close'):
                await self.retrieval_agent.close()
        except Exception as e:
            logger.warning(f"关闭客户端时发生错误: {e}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def _retrieve_selection(self, query: str,  doc_ids: List[str] = []) -> List[dict]:
        """检索并筛选相关文本块。
        
        根据查询和文档ID列表检索相关的文本块，并通过智能筛选
        确保返回的内容与查询高度相关。
        
        Args:
            query: 检索查询字符串
            doc_ids: 目标文档ID列表，用于限制检索范围
            
        Returns:
            筛选后的相关文本块列表，每个元素包含文本内容和元数据
            
        处理流程：
            1. 调用检索客户端获取候选文本块
            2. 使用智能筛选算法评估相关性
            3. 返回高相关性的文本块列表
        """
        
        chunks_response = await self.retrieval_agent.retrieve_chunks_async(
            query=query,
            doc_ids=doc_ids
        )
        chunks_result = chunks_response.results 
        logger.info(f"Retrieved {len(chunks_result)} chunks for query: {query}")
        # 根据chunk_id进行去重
        unique_chunks = {chunk['chunk_id']: chunk for chunk in chunks_result}
        chunks_result = list(unique_chunks.values())
        logger.info(f"After deduplication, {len(chunks_result)} unique chunks remain")    
        
        selected_chunks = await make_decision_of_chunk_retrieval(
            query_text=query,
            chunks=chunks_result
        ) 
        logger.info(f"Selected {len(selected_chunks)} chunks after filtering")
        
        return selected_chunks
        
    @workflow(
        name="RAG Agent"
    )
    async def rag_agent(self, query: str, dataset_type: str = "llm_papers") -> Dict[str, Any]:
        """RAG智能代理主处理函数。
        
        这是RAG系统的核心入口函数，整合了意图识别、文档检索、
        查询改写、内容检索和答案生成的完整流程。
        
        Args:
            query: 用户输入的问题
            dataset_type: 数据集类型，默认为"llm_papers"
            
        Returns:
            包含以下信息的字典：
            - answer: 生成的答案字符串
            - chunk_ids: 使用的chunk ID列表
            - cost_breakdown: 费用明细
            - performance_metrics: 性能指标
            
        处理流程：
            1. 意图识别：判断问题是事实型还是信息汇总型
            2. 文档检索：根据数据集类型检索相关文档
            3. 查询改写：基于意图和文档摘要生成多个改写查询
            4. 内容检索：并行执行多个改写查询的检索
            5. 答案生成：基于所有检索结果生成最终答案
            
        性能监控：
            - 记录每个步骤的处理时间
            - 输出详细的日志信息
            - 统计总体处理时间
            - 记录费用信息
            
        异常处理：
            - 如果任何步骤失败，会抛出相应异常
            - 确保资源正确释放
        """ 
        start_time = time()
        logger.info(f"RAG Agent is processing query: {query}")
        
        # 初始化费用统计
        cost_breakdown = {
            "intent_recognition_cost": 0,
            "query_rewrite_cost": 0,
            "generation_cost": 0,
            "total_cost_usd": 0
        }
        
        documents_response = await self.retrieval_agent.retrieve_documents_by_dataset_async(
            data_set_type=dataset_type,
            query=query
        )
        documents_result = documents_response.results
        doc_ids = []
        doc_summaries = []
        for doc in documents_result:
            doc_id = doc.get("file_id")
            doc_summary = doc.get("summary")
            if doc_id:
                doc_ids.append(doc_id)
                doc_summaries.append(doc_summary)
        logger.info(f"Documents are retrieved: {json.dumps(doc_ids, indent=4, ensure_ascii=False)}")
        documents_retrieval_cost = time() - start_time
        logger.info(f"Documents retrieval cost: {documents_retrieval_cost} seconds")
        
        documents_summary = "\n".join(doc_summaries)
        
        # Step 1: Query Rewriting
        rewritten_queries, query_rewrite_cost_info = await self.query_rewrite_agent.rewrite_query(query, documents_summary)
        logger.info(f"Rewritten queries are generated: {json.dumps([query.rewritten_query for query in rewritten_queries], indent=4, ensure_ascii=False)}")
        query_rewrite_cost = time() - start_time - documents_retrieval_cost
        logger.info(f"Query rewrite cost: {query_rewrite_cost} seconds") 
        
        # 记录查询改写的费用
        if isinstance(query_rewrite_cost_info, dict) and "total_cost" in query_rewrite_cost_info:
            cost_breakdown["query_rewrite_cost"] = query_rewrite_cost_info.get("total_cost", 0)
        
        # Step 2: Retrieval
        tasks = [self._retrieve_selection(rewritten_query.rewritten_query, doc_ids) for rewritten_query in rewritten_queries]
        retrieval_results = await asyncio.gather(*tasks)
        all_reference = [] # type: ignore
        for retrieval_result in retrieval_results:
            all_reference.extend(retrieval_result)
        logger.info(f"Number of retrieved chunks: {len(all_reference)}")
        if all_reference:
            logger.info(f"Example of retrieved chunks: {json.dumps(all_reference[0], indent=4, ensure_ascii=False)}")
        else:
            logger.warning("No chunks were retrieved")
        
        retrieval_cost = time() - start_time - documents_retrieval_cost - query_rewrite_cost
        logger.info(f"Retrieval cost: {retrieval_cost} seconds")
        
        # Step 3: Generation
        logger.info(f"len(all_reference): {len(all_reference)}")
        # 使用chunk_id 再过滤一次，确保每个chunk_id只保留一个
        unique_references = {chunk['chunk_id']: chunk for chunk in all_reference}
        all_reference = list(unique_references.values())
        logger.info(f"After deduplication, {len(all_reference)} unique chunks remain")
        
        # 提取chunk IDs
        chunk_ids = [chunk.get('chunk_id', '') for chunk in all_reference if chunk.get('chunk_id')]
        
        intent_flag = rewritten_queries[0].intent_flag
        final_answer, generation_cost_info = await self.generation_agent.generate_answer(query, intent_flag, all_reference)
        logger.info(f"Final answer is generated: {final_answer}")
        generation_cost = time() - start_time - documents_retrieval_cost - query_rewrite_cost - retrieval_cost
        logger.info(f"Generation cost: {generation_cost} seconds")
        
        # 记录答案生成的费用
        if isinstance(generation_cost_info, dict) and "total_cost" in generation_cost_info:
            cost_breakdown["generation_cost"] = generation_cost_info.get("total_cost", 0)
        
        total_time = time() - start_time
        logger.info(f"Total cost: {total_time} seconds")
        
        # 计算总费用
        cost_breakdown["total_cost_usd"] = (
            cost_breakdown["intent_recognition_cost"] +
            cost_breakdown["query_rewrite_cost"] +
            cost_breakdown["generation_cost"]
        )
        
        # 构建返回结果
        result = {
            "answer": final_answer,
            "chunk_ids": chunk_ids,
            "cost_breakdown": cost_breakdown,
            "performance_metrics": {
                "total_time_seconds": total_time,
                "documents_retrieval_time": documents_retrieval_cost,
                "query_rewrite_time": query_rewrite_cost,
                "retrieval_time": retrieval_cost,
                "generation_time": generation_cost
            },
            "metadata": {
                "query": query,
                "dataset_type": dataset_type,
                "intent_flag": intent_flag,
                "num_chunks_used": len(chunk_ids),
                "num_documents_retrieved": len(doc_ids)
            }
        }
        
        return result
        
