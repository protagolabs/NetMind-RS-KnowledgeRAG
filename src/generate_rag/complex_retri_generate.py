#!/usr/bin/env python3
"""
复杂检索生成脚本

基于深度研究脚本，实现四步检索生成流程：
1. Step1: 意图识别 - 判断查询是信息汇总型(1)还是事实型(0)
2. Step2: 查询改写 - 基于文档总结改写查询以增加召回
3. Step3: 检索 - 使用API检索多个查询
4. Step4: 生成 - 基于检索结果生成最终答案

Author: yujing.wang
Date: 2025.01.27
"""

import json
import time
import requests
import os
import logging
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field
import openai
from openai import AsyncOpenAI
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# OpenAI Cost Calculator
# ============================================================================

class OpenAICostCalculator:
    """OpenAI费用计算器。
    
    根据GPT-4.1的收费标准计算费用：
    - 输入：每1M tokens $2.00
    - 输出：每1M tokens $8.00
    """
    
    def __init__(self):
        """初始化费用计算器。"""
        # GPT-4.1 收费标准 (每1M tokens)
        self.input_cost_per_1m_tokens = 2.00  # $2.00 per 1M input tokens
        self.output_cost_per_1m_tokens = 8.00  # $8.00 per 1M output tokens
        
        # 转换为每token的费用
        self.input_cost_per_token = self.input_cost_per_1m_tokens / 1_000_000
        self.output_cost_per_token = self.output_cost_per_1m_tokens / 1_000_000
        
        # 初始化tokenizer
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"无法初始化tiktoken编码器: {e}")
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量。
        
        Args:
            text: 要计算的文本
            
        Returns:
            token数量
        """
        if not text:
            return 0
        
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"Token计算失败: {e}")
                # 使用简单的字符估算作为后备
                return len(text) // 4
        else:
            # 使用简单的字符估算作为后备
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算消息列表的token数量。
        
        Args:
            messages: 消息列表
            
        Returns:
            token数量
        """
        if not messages:
            return 0
        
        total_tokens = 0
        
        for message in messages:
            # 每个消息的基础token
            total_tokens += 3  # 基础消息token
            
            # 计算内容的token
            content = message.get('content', '')
            if content:
                total_tokens += self.count_tokens(content)
            
            # 如果有name字段，额外token
            if 'name' in message:
                total_tokens += 1
        
        return total_tokens
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """计算费用。
        
        Args:
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            
        Returns:
            包含各项费用的字典
        """
        input_cost = input_tokens * self.input_cost_per_token
        output_cost = output_tokens * self.output_cost_per_token
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6)
        }
    
    def calculate_messages_cost(self, messages: List[Dict[str, str]], completion_text: str = "") -> Dict[str, float]:
        """计算消息和完成文本的费用。
        
        Args:
            messages: 消息列表
            completion_text: 完成文本
            
        Returns:
            包含各项费用的字典
        """
        input_tokens = self.count_messages_tokens(messages)
        output_tokens = self.count_tokens(completion_text) if completion_text else 0
        
        return self.calculate_cost(input_tokens, output_tokens)


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class IntentRecognitionResult(BaseModel):
    """意图识别结果模型。
    
    Attributes:
        intent_type: 意图类型 (0: 事实型, 1: 信息汇总型)
        reasoning: 推理过程
    """
    intent_type: int = Field(..., ge=0, le=1, description="意图类型：0=事实型，1=信息汇总型")
    reasoning: str = Field(..., description="推理过程")


class QueryRewriteItem(BaseModel):
    """查询改写项模型。
    
    Attributes:
        rewritten_query: 改写后的查询
        reasoning: 改写理由
    """
    rewritten_query: str = Field(..., description="改写后的查询")
    reasoning: str = Field(..., description="改写理由")


class QueryRewriteResult(BaseModel):
    """查询改写结果模型。
    
    Attributes:
        queries: 改写后的查询列表
    """
    queries: List[QueryRewriteItem] = Field(..., description="改写后的查询列表")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ComplexRetriConfig:
    """复杂检索配置类。
    
    Attributes:
        rag_api_url: 检索API的URL
        model_name: 使用的AI模型名称
        temperature: AI模型温度参数
        max_tokens: 最大token数
        max_queries: 最大查询数量
        knowledge_rag_api_url: KnowledgeRAG API的URL
    """
    rag_api_url: str = "http://localhost:8001"
    knowledge_rag_api_url: str = "http://71.178.110.3:8955"
    model_name: str = "gpt-4.1"
    temperature: float = 0.1
    max_tokens: int = 4000
    max_queries: int = 5


@dataclass
class QueryIntent:
    """查询意图识别结果。
    
    Attributes:
        query: 原始查询
        intent_type: 意图类型 (0: 事实型, 1: 信息汇总型)
        reasoning: 推理过程
    """
    query: str
    intent_type: int
    reasoning: str


@dataclass
class RewrittenQuery:
    """改写后的查询。
    
    Attributes:
        original_query: 原始查询
        rewritten_query: 改写后的查询
        intent_type: 意图类型
        reasoning: 改写理由
    """
    original_query: str
    rewritten_query: str
    intent_type: int
    reasoning: str


@dataclass
class RetrievalResult:
    """检索结果。
    
    Attributes:
        query: 查询
        success: 是否成功
        data: 检索数据
        error: 错误信息
        doc_ids: 文档ID列表
        chunks: chunk列表
        decisions: 决策结果列表
    """
    query: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    doc_ids: List[str] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProcessStep:
    """处理步骤记录。
    
    Attributes:
        step_name: 步骤名称
        start_time: 开始时间
        end_time: 结束时间
        duration: 持续时间
        result: 步骤结果
        error: 错误信息
    """
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ============================================================================
# KnowledgeRAG API Client
# ============================================================================

class KnowledgeRAGAPIClient:
    """KnowledgeRAG API客户端。
    
    提供统一的API调用接口，支持三个核心功能：
    1. 文档检索
    2. Chunk检索
    3. Chunk决策
    """
    
    def __init__(self, api_base_url: str = "http://71.178.110.3:8955"):
        """初始化API客户端。
        
        Args:
            api_base_url: API基础URL
        """
        self.api_base_url = api_base_url
        self.client = None
    
    async def _ensure_client(self):
        """确保客户端已初始化。"""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """关闭客户端。"""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def retrieve_documents(self, query: str, dataset_type: str = "llm_papers") -> Dict[str, Any]:
        """文档检索 - 获取文档summary和doc_ids。
        
        Args:
            query: 查询文本
            dataset_type: 数据集类型
            
        Returns:
            文档检索结果
        """
        await self._ensure_client()
        
        endpoint = "/doc-retrieval-by-dataset"
        url = f"{self.api_base_url}{endpoint}"
        
        payload = {
            "data_set_type": dataset_type,
            "query": query
        }
        
        try:
            logger.info(f"开始文档检索: {query}")
            response = await self.client.post(url, json=payload)
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
            
            response_data = response.json()
            
            if not response_data.get('success', False):
                return {
                    "success": False,
                    "error": response_data.get('message', 'API返回失败')
                }
            
            # 解析结果
            results = response_data.get('results', [])
            doc_ids = []
            summaries = []
            
            for doc in results:
                doc_id = doc.get('file_id')
                summary = doc.get('summary', '')
                if doc_id:
                    doc_ids.append(doc_id)
                    summaries.append(summary)
            
            return {
                "success": True,
                "doc_ids": doc_ids,
                "summaries": summaries,
                "total_results": len(results),
                "response_data": response_data
            }
            
        except Exception as e:
            logger.error(f"文档检索异常: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def retrieve_chunks(self, query: str, doc_ids: List[str], each_doc_chunk_number: int = 10) -> Dict[str, Any]:
        """Chunk检索 - 获取相关chunks。
        
        Args:
            query: 查询文本
            doc_ids: 文档ID列表
            each_doc_chunk_number: 每个文档的chunk数量
            
        Returns:
            Chunk检索结果
        """
        await self._ensure_client()
        
        endpoint = "/chunk-retrieval"
        url = f"{self.api_base_url}{endpoint}"
        
        payload = {
            "query": query,
            "doc_ids": doc_ids,
            "each_doc_chunk_number": each_doc_chunk_number
        }
        
        try:
            logger.info(f"开始Chunk检索: {query}, 文档数: {len(doc_ids)}")
            response = await self.client.post(url, json=payload)
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
            
            response_data = response.json()
            
            if not response_data.get('success', False):
                return {
                    "success": False,
                    "error": response_data.get('message', 'API返回失败')
                }
            
            # 解析结果
            results = response_data.get('results', [])
            chunks = []
            
            for chunk in results:
                chunk_data = {
                    "chunk_id": chunk.get('chunk_id'),
                    "summary": chunk.get('summary', ''),
                    "insights": chunk.get('insights', ''),
                    "key_words": chunk.get('key_words', []),
                }
                chunks.append(chunk_data)
            
            return {
                "success": True,
                "chunks": chunks,
                "total_results": len(chunks),
                "response_data": response_data
            }
            
        except Exception as e:
            logger.error(f"Chunk检索异常: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def make_chunk_decisions(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Chunk决策 - 对chunks进行决策匹配。
        
        Args:
            query: 查询文本
            chunks: chunk列表
            
        Returns:
            Chunk决策结果
        """
        await self._ensure_client()
        
        endpoint = "/chunk-decision"
        url = f"{self.api_base_url}{endpoint}"
        
        payload = {
            "query_text": query,
            "chunks": chunks
        }
        
        try:
            logger.info(f"开始Chunk决策: {query}, chunk数: {len(chunks)}")
            response = await self.client.post(url, json=payload)
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
            
            response_data = response.json()
            
            if not response_data.get('success', False):
                return {
                    "success": False,
                    "error": response_data.get('message', 'API返回失败')
                }
            
            # 解析结果
            results = response_data.get('results', [])
            decisions = []
            
            for decision in results:
                decision_data = {
                    "chunk_id": decision.get('chunk_id'),
                    "insights": decision.get('insights', ''),
                    "key_words": decision.get('key_words', []),
                    "summary": decision.get('summary', ''),
                }
                decisions.append(decision_data)
            
            return {
                "success": True,
                "decisions": decisions,
                "total_results": len(decisions),
                "response_data": response_data
            }
            
        except Exception as e:
            logger.error(f"Chunk决策异常: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# OpenAI Client Wrapper
# ============================================================================

class OpenAIWrapper:
    """OpenAI客户端包装类。
    
    提供统一的OpenAI API调用接口，支持同步和异步调用。
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """初始化OpenAI包装器。
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要设置 OPENAI_API_KEY 环境变量")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.cost_calculator = OpenAICostCalculator()
    
    def _handle_openai_error(self, error: Exception, operation: str) -> str:
        """处理OpenAI API错误。
        
        Args:
            error: 错误对象
            operation: 操作名称
            
        Returns:
            错误信息字符串
        """
        if "rate_limit" in str(error).lower():
            return f"{operation} - OpenAI API速率限制，请稍后重试"
        elif "quota" in str(error).lower():
            return f"{operation} - OpenAI API配额已用完"
        elif "authentication" in str(error).lower():
            return f"{operation} - OpenAI API认证失败，请检查API密钥"
        else:
            return f"{operation} - OpenAI API错误: {error}"
    
    async def parse_structured_response(
        self, 
        model: str,
        messages: List[Dict[str, str]],
        response_model: BaseModel,
        temperature: float = 0.1
    ) -> Optional[Tuple[BaseModel, Dict[str, float]]]:
        """异步解析结构化响应。
        
        Args:
            model: 模型名称
            messages: 消息列表
            response_model: 响应模型类
            temperature: 温度参数
            
        Returns:
            解析后的模型对象和费用信息的元组，如果失败则返回None
        """
        try:
            completion = await self.client.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_model
            )
            
            # 计算费用
            completion_text = completion.choices[0].message.content or ""
            cost_info = self.calculate_cost(messages, completion_text)
            
            return completion.choices[0].message.parsed, cost_info
        except Exception as e:
            error_msg = self._handle_openai_error(e, "结构化响应解析")
            logger.error(error_msg)
            return None
    
    async def generate_text(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """异步生成文本。
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            生成的文本和费用信息的元组，如果失败则返回None
        """
        try:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 1500
            )
            
            completion_text = completion.choices[0].message.content or ""
            
            # 计算费用
            cost_info = self.calculate_cost(messages, completion_text)
            
            return completion_text, cost_info
        except Exception as e:
            error_msg = self._handle_openai_error(e, "文本生成")
            logger.error(error_msg)
            return None
    
    def calculate_cost(self, messages: List[Dict[str, str]], completion_text: str = "") -> Dict[str, float]:
        """计算OpenAI API调用的费用。
        
        Args:
            messages: 消息列表
            completion_text: 完成文本
            
        Returns:
            包含各项费用的字典
        """
        return self.cost_calculator.calculate_messages_cost(messages, completion_text)


# ============================================================================
# Step Implementation Class
# ============================================================================

class ComplexRetriSteps:
    """复杂检索步骤实现类。
    
    实现四个核心步骤：意图识别、查询改写、检索、生成。
    每个方法都是独立的，便于测试和调试。
    """
    
    def __init__(self, config: ComplexRetriConfig):
        """初始化步骤实现类。
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.openai_wrapper = OpenAIWrapper()
        self.knowledge_rag_client = None
    
    def check_api_health(self) -> bool:
        """检查API健康状态。
        
        Returns:
            API是否健康可用
        """
        try:
            response = requests.get(f"{self.config.knowledge_rag_api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"API健康检查通过: {health_data.get('documents_count', 0)} 个文档")
                return True
            else:
                logger.error(f"API健康检查失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"API健康检查错误: {e}")
            return False
    
    async def get_documents_summary_via_api(self, query: str, dataset_type: str = "llm_papers") -> Dict[str, Any]:
        """通过API获取文档总结。
        
        Args:
            query: 查询文本
            dataset_type: 数据集类型
            
        Returns:
            文档检索结果
        """
        if self.knowledge_rag_client is None:
            self.knowledge_rag_client = KnowledgeRAGAPIClient(self.config.knowledge_rag_api_url)
        
        try:
            result = await self.knowledge_rag_client.retrieve_documents(query, dataset_type)
            
            if result.get("success"):
                summaries = result.get("summaries", [])
                if summaries:
                    return {
                        "doc_ids": result.get("doc_ids", []),
                        "summaries": "\n".join(summaries)
                    }
                else:
                    return {
                        "doc_ids": result.get("doc_ids", []),
                        "summaries": "无文档总结信息"
                    }
            else:
                logger.error(f"获取文档总结失败: {result.get('error')}")
                return {
                    "doc_ids": [],
                    "summaries": "获取文档总结失败"
                }
        finally:
            # 确保在完成后关闭客户端
            if self.knowledge_rag_client is not None:
                await self.knowledge_rag_client.close()
    
    def get_documents_summary(self) -> str:
        """获取所有文档的总结（原有方法，保留兼容性）。
        
        Returns:
            文档总结文本
        """
        try:
            response = requests.get(f"{self.config.rag_api_url}/documents/summary", timeout=30)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return result.get("data", {}).get("summary", "无文档总结信息")
                else:
                    return "获取文档总结失败"
            else:
                return f"获取文档总结失败: HTTP {response.status_code}"
        except Exception as e:
            logger.error(f"获取文档总结错误: {e}")
            return f"获取文档总结错误: {e}"
    
    async def step1_intent_recognition(self, query: str) -> Optional[Tuple[QueryIntent, Dict[str, float]]]:
        """步骤1: 意图识别。
        
        Args:
            query: 查询文本
            
        Returns:
            意图识别结果和费用信息的元组，如果失败则返回None
        """
        try:
            prompt = f"""You are a master of intent recognition. Please determine whether the "User Question" involves the intent of "concept listing", such as < summary type questions >, etc. If it does, output 1; otherwise, output 0.

Example:
[User Question] : Based on the inpatient medical records of Zhuque Central Hospital, summarize the diagnostic basis and differential diagnosis of Ge Moumou.
Output: 1

[User question] : {query}
Output format: 1 or 0
"""

            messages = [{"role": "user", "content": prompt}]
            
            result = await self.openai_wrapper.parse_structured_response(
                model=self.config.model_name,
                messages=messages,
                response_model=IntentRecognitionResult,
                temperature=0.1
            )
            
            if result is None:
                return None
            
            parsed_result, cost_info = result
            
            return QueryIntent(
                query=query,
                intent_type=parsed_result.intent_type,
                reasoning=parsed_result.reasoning
            ), cost_info
            
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return None
    
    async def step2_query_rewriting(
        self, 
        query: str, 
        intent: QueryIntent, 
        documents_summary: str
    ) -> Optional[Tuple[List[RewrittenQuery], Dict[str, float]]]:
        """步骤2: 查询改写。
        
        Args:
            query: 原始查询
            intent: 意图识别结果
            documents_summary: 文档总结
            
        Returns:
            改写后的查询列表和费用信息的元组，如果失败则返回None
        """
        try:
            intent_description = "信息汇总型" if intent.intent_type == 1 else "事实型"
            
            prompt = f"""You are an expert in information retrieval and query rewriting.

Given:
- The original user query: "{query}"
- The query type: {intent_description}
- The reasoning/intention behind the query: {intent.reasoning}
- A summary (or keywords) of all documents in the current database: {documents_summary}

Your task:
1. Based on the query type and intent, rewrite the original query into multiple alternative versions, each approaching the information need from a different angle.
2. For information aggregation (summarization/listing) queries, ensure the rewrites cover a wide range of relevant aspects and perspectives from the document summaries.
3. For fact-based queries, make each rewrite target specific, clearly identifiable information in the database.
4. Each rewritten query must have a clear retrieval focus and be formulated to maximize the recall of relevant information in the database, making explicit use of document summaries or keywords as reference.
5. Generate up to {self.config.max_queries} diverse rewritten queries.

Instructions:
- Use the original query and the summaries/keywords of all database documents together as context to inform your rewrites.
- Leverage your understanding of the query type and intention to tailor each rewrite for optimal information coverage and retrieval precision.
- Output each rewritten query as a separate item in a numbered list.

Please generate the rewritten queries now.
"""

            messages = [{"role": "user", "content": prompt}]
            
            result = await self.openai_wrapper.parse_structured_response(
                model=self.config.model_name,
                messages=messages,
                response_model=QueryRewriteResult,
                temperature=0.2
            )
            
            if result is None:
                return None
            
            parsed_result, cost_info = result
            
            rewritten_queries = []
            for query_item in parsed_result.queries:
                rewritten_query = RewrittenQuery(
                    original_query=query,
                    rewritten_query=query_item.rewritten_query,
                    intent_type=intent.intent_type,
                    reasoning=query_item.reasoning
                )
                rewritten_queries.append(rewritten_query)
            
            return rewritten_queries, cost_info
            
        except Exception as e:
            logger.error(f"查询改写失败: {e}")
            return None
    
    async def step3_retrieval(self, queries: List[RewrittenQuery], doc_ids: List[str]) -> List[RetrievalResult]:
        """步骤3: 检索 - 使用KnowledgeRAG API进行检索。
        
        Args:
            queries: 改写后的查询列表
            doc_ids: 文档ID列表
            
        Returns:
            检索结果列表
        """
        retrieval_results = []
        
        if self.knowledge_rag_client is None:
            self.knowledge_rag_client = KnowledgeRAGAPIClient(self.config.knowledge_rag_api_url)
        
        try:
            for query_obj in queries:
                try:
                    logger.info(f"检索查询: {query_obj.rewritten_query}")
                    
                    # 第一步：Chunk检索
                    chunk_result = await self.knowledge_rag_client.retrieve_chunks(query_obj.rewritten_query, doc_ids)
                    if not chunk_result.get("success"):
                        result = RetrievalResult(
                            query=query_obj.rewritten_query,
                            success=False,
                            error=f"Chunk检索失败: {chunk_result.get('error')}"
                        )
                        retrieval_results.append(result)
                        continue
                    
                    chunks = chunk_result.get("chunks", [])
                    if not chunks:
                        result = RetrievalResult(
                            query=query_obj.rewritten_query,
                            success=False,
                            error="未找到相关chunks"
                        )
                        retrieval_results.append(result)
                        continue
                    
                    # 第二步：Chunk决策
                    decision_result = await self.knowledge_rag_client.make_chunk_decisions(query_obj.rewritten_query, chunks)
                    if not decision_result.get("success"):
                        result = RetrievalResult(
                            query=query_obj.rewritten_query,
                            success=False,
                            error=f"Chunk决策失败: {decision_result.get('error')}"
                        )
                        retrieval_results.append(result)
                        continue
                    
                    decisions = decision_result.get("decisions", [])
                    
                    # 构建检索结果
                    result = RetrievalResult(
                        query=query_obj.rewritten_query,
                        success=True,
                        doc_ids=doc_ids,
                        chunks=chunks,
                        decisions=decisions,
                        data={
                            "doc_ids": doc_ids,
                            "chunk_result": chunk_result,
                            "decision_result": decision_result
                        }
                    )
                    
                    retrieval_results.append(result)
                    
                except Exception as e:
                    logger.error(f"检索查询 '{query_obj.rewritten_query}' 时出错: {e}")
                    result = RetrievalResult(
                        query=query_obj.rewritten_query,
                        success=False,
                        error=str(e)
                    )
                    retrieval_results.append(result)
        
        finally:
            # 确保在所有查询完成后关闭客户端
            if self.knowledge_rag_client is not None:
                await self.knowledge_rag_client.close()
        
        return retrieval_results
    
    async def step4_generation(
        self,
        original_query: str,
        intent: QueryIntent,
        retrieval_results: List[RetrievalResult],
        documents_result: dict
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """步骤4: 生成最终答案。
        
        Args:
            original_query: 原始查询
            intent: 意图识别结果
            retrieval_results: 检索结果列表
            documents_result: 文档检索结果
            
        Returns:
            生成的最终答案和费用信息的元组，如果失败则返回None
        """
        try:
            # 收集所有成功的检索结果
            successful_results = [r for r in retrieval_results if r.success]
            
            if not successful_results:
                return "未找到相关信息", {}
            
            # 格式化检索结果
            retrieval_text = ""
            all_relevant_decisions = []
            
            # 收集所有成功的检索结果中的相关决策
            for result in successful_results:
                relevant_decisions = [d for d in result.decisions]
                all_relevant_decisions.extend(relevant_decisions)
            print(f"总共找到 {len(all_relevant_decisions)} 个相关决策")
            
            if all_relevant_decisions:
                retrieval_text += f"### Related retrieval content:\n"
                for j, decision in enumerate(all_relevant_decisions, 1):
                    retrieval_text += f"**Retrieval content {j}**:\n"
                    retrieval_text += f"Content: {decision.get('insights', '')}\n\n"
            else:
                retrieval_text += f"### No related retrieval content\n\n"
            
            retrieval_text += "---\n\n"
            
            # 添加文档总结信息
            documents_summary = documents_result.get("summaries", "无文档总结信息")
            if isinstance(documents_summary, list):
                documents_summary = "\n".join(documents_summary)
            retrieval_text += f"### Document Summaries:\n{documents_summary}\n\n"
            

            intent_description = "information aggregation" if intent.intent_type == 1 else "fact-based"

            prompt = f"""# Structured Answer

You are a professional AI assistant. Please answer the user's question strictly based on the retrieved content below.  
**For every key statement, fact, or summary you write, you must add a reference using the special format:**  
`<reference>{{reference_order_number}}</reference>`  
where `reference_order_number` corresponds to the order of the source in the References section at the end.

**User's original question:**  
`{original_query}`

**Question type:**  
`{intent_description}`

**Question intent:**  
`{intent.reasoning}`

**Retrieved content (from multiple chunks and documents):**  
{retrieval_text}  

---

## Instructions

1. Carefully read and analyze all retrieved content, ensuring **no relevant information is missed or omitted**.
2. Before writing your answer, plan and structure the content logically—organize by theme, topic, or other suitable dimensions.  
   - Do **not** simply list or repeat chunks; instead, synthesize and integrate information across chunks.
   - Use clear section headings (##, ###), bullet points, or numbered lists as needed.
3. For information aggregation questions (type = 1):
   - Provide a comprehensive, well-organized summary, grouped by topic, theme, or logic.
   - Highlight key points and important trends.
   - For every key point, include the appropriate `<reference>{{n}}</reference>` marker.
4. For fact-based questions (type = 0):
   - Provide a precise, direct answer.
   - For every statement, fact, or piece of evidence, include the corresponding `<reference>{{n}}</reference>` marker.
   - If there are multiple facts, list and organize them clearly by topic or logic.
5. **General requirements:**
   - Do **not** introduce or invent any information that is not present in the retrieved chunks or document summaries.
   - Make sure your answer is accurate, complete, and well-structured.
   - Use only the special `<reference>{{n}}</reference>` format for all citations.
   - The same chunk can be referenced multiple times as needed.
   - Each referenced number must correspond to a source listed in the References section at the end.
   - Ensure your writing is logical, objective, and easy to read.

---

## Output format

- Start with a brief summary statement.
- Organize the main body using markdown section headings (##, ###), bullet points, or numbered lists.
- For **every sentence, claim, or important fact**, add `<reference>{{n}}</reference>` after it.
- Do not cite specific paragraph or line positions—only use the provided order numbers for each chunk.

"""

            messages = [{"role": "user", "content": prompt}]
            
            result = await self.openai_wrapper.generate_text(
                model=self.config.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=self.config.max_tokens
            )
            
            if result is None:
                return None, {}
            
            final_answer, cost_info = result
            return final_answer, cost_info
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return None, {}


# ============================================================================
# Workflow Control Class
# ============================================================================

class ComplexRetriWorkflow:
    """复杂检索工作流控制类。
    
    整合所有步骤，提供端到端的工作流程控制。
    包含状态管理、错误处理、进度跟踪等功能。
    """
    
    def __init__(self, config: Optional[ComplexRetriConfig] = None):
        """初始化工作流控制类。
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or ComplexRetriConfig()
        self.steps = ComplexRetriSteps(self.config)
        self.process_steps: List[ProcessStep] = []
        
        # 创建结果目录
        self.result_dir = "agentic_rag_md"
        self.rag_result_dir = "rag_result"
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.rag_result_dir, exist_ok=True)
    
    def _start_step(self, step_name: str) -> ProcessStep:
        """开始一个处理步骤。
        
        Args:
            step_name: 步骤名称
            
        Returns:
            步骤记录对象
        """
        step = ProcessStep(
            step_name=step_name,
            start_time=time.time()
        )
        self.process_steps.append(step)
        logger.info(f"开始步骤: {step_name}")
        return step
    
    def _end_step(self, step: ProcessStep, result: Optional[Dict[str, Any]] = None, 
                  error: Optional[str] = None):
        """结束一个处理步骤。
        
        Args:
            step: 步骤记录对象
            result: 步骤结果
            error: 错误信息
        """
        step.end_time = time.time()
        step.duration = step.end_time - step.start_time
        step.result = result
        step.error = error
        
        if error:
            logger.error(f"步骤 {step.step_name} 失败: {error}")
        else:
            logger.info(f"步骤 {step.step_name} 完成，耗时: {step.duration:.2f}秒")
    
    def save_markdown_result(
        self, 
        original_query: str, 
        intent: QueryIntent, 
        rewritten_queries: List[RewrittenQuery], 
        retrieval_results: List[RetrievalResult], 
        final_answer: str,
        step_costs: Dict[str, Dict[str, float]],
        total_cost: float
    ) -> str:
        """保存结果为Markdown格式。
        
        Args:
            original_query: 原始查询
            intent: 意图识别结果
            rewritten_queries: 改写后的查询列表
            retrieval_results: 检索结果列表
            final_answer: 最终答案
            step_costs: 各步骤的费用信息
            total_cost: 总费用
            
        Returns:
            保存的文件路径
        """
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.md"
        filepath = os.path.join(self.result_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 复杂检索生成结果\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 原始查询
            f.write("## 📝 原始查询\n\n")
            f.write(f"**查询**: {original_query}\n\n")
            f.write(f"**意图类型**: {'信息汇总型' if intent.intent_type == 1 else '事实型'}\n\n")
            f.write(f"**推理过程**: {intent.reasoning}\n\n")
            
            # 费用统计
            f.write("## 💰 费用统计\n\n")
            f.write(f"**总费用**: ${total_cost:.6f} USD\n\n")
            f.write("**各步骤费用**:\n")
            for step_name, cost_info in step_costs.items():
                step_cost = cost_info.get("total_cost_usd", 0)
                input_tokens = cost_info.get("input_tokens", 0)
                output_tokens = cost_info.get("output_tokens", 0)
                f.write(f"- **{step_name}**: ${step_cost:.6f} (输入: {input_tokens} tokens, 输出: {output_tokens} tokens)\n")
            f.write("\n")
            
            # 处理步骤统计
            f.write("## ⏱️ 处理步骤统计\n\n")
            total_time = sum(step.duration or 0 for step in self.process_steps)
            f.write(f"**总处理时间**: {total_time:.2f} 秒\n\n")
            
            for step in self.process_steps:
                status = "✅ 成功" if not step.error else "❌ 失败"
                f.write(f"- **{step.step_name}**: {status} ({step.duration:.2f}秒)")
                if step.result and "cost_info" in step.result:
                    cost_info = step.result["cost_info"]
                    step_cost = cost_info.get("total_cost_usd", 0)
                    f.write(f" - 费用: ${step_cost:.6f}")
                f.write("\n")
                if step.error:
                    f.write(f"  - 错误: {step.error}\n")
            f.write("\n")
            
            # 查询改写
            f.write("## 🔄 查询改写\n\n")
            for i, query in enumerate(rewritten_queries, 1):
                f.write(f"### 改写查询 {i}\n\n")
                f.write(f"**原始查询**: {query.original_query}\n\n")
                f.write(f"**改写查询**: {query.rewritten_query}\n\n")
                f.write(f"**改写理由**: {query.reasoning}\n\n")
            f.write("\n")
            
            # 检索结果
            f.write("## 🔍 检索结果\n\n")
            success_count = sum(1 for r in retrieval_results if r.success)
            f.write(f"**检索成功率**: {success_count}/{len(retrieval_results)} ({success_count/len(retrieval_results)*100:.1f}%)\n\n")
            
            for i, result in enumerate(retrieval_results, 1):
                f.write(f"### 检索结果 {i}\n\n")
                f.write(f"**查询**: {result.query}\n\n")
                f.write(f"**状态**: {'✅ 成功' if result.success else '❌ 失败'}\n\n")
                
                if result.success:
                    f.write(f"**文档ID数量**: {len(result.doc_ids)}\n\n")
                    f.write(f"**Chunk数量**: {len(result.chunks)}\n\n")
                    f.write(f"**决策数量**: {len(result.decisions)}\n\n")
                    
                    # 显示相关决策
                    relevant_decisions = [d for d in result.decisions]
                    f.write(f"**相关决策数量**: {len(relevant_decisions)}\n\n")
                    
                    if relevant_decisions:
                        f.write("**相关决策**:\n")
                        for j, decision in enumerate(relevant_decisions[:3], 1):  # 只显示前3个
                            f.write(f"- {decision.get('chunk_id', 'Unknown')}: {decision.get('relevance_score', 0):.3f}\n")
                        f.write("\n")
                elif result.error:
                    f.write(f"**错误**: {result.error}\n\n")
                
                f.write("---\n\n")
            
            # 最终答案
            f.write("## 📋 最终答案\n\n")
            f.write(final_answer)
            f.write("\n\n")
            
            # 处理步骤详情
            f.write("## 🔧 处理步骤详情\n\n")
            for step in self.process_steps:
                f.write(f"### {step.step_name}\n\n")
                f.write(f"**开始时间**: {datetime.fromtimestamp(step.start_time).strftime('%H:%M:%S')}\n\n")
                f.write(f"**结束时间**: {datetime.fromtimestamp(step.end_time or step.start_time).strftime('%H:%M:%S')}\n\n")
                f.write(f"**持续时间**: {step.duration:.2f} 秒\n\n")
                
                if step.result and "cost_info" in step.result:
                    cost_info = step.result["cost_info"]
                    f.write("**费用信息**:\n")
                    f.write(f"- 输入tokens: {cost_info.get('input_tokens', 0)}\n")
                    f.write(f"- 输出tokens: {cost_info.get('output_tokens', 0)}\n")
                    f.write(f"- 输入费用: ${cost_info.get('input_cost_usd', 0):.6f}\n")
                    f.write(f"- 输出费用: ${cost_info.get('output_cost_usd', 0):.6f}\n")
                    f.write(f"- 总费用: ${cost_info.get('total_cost_usd', 0):.6f}\n\n")
                
                if step.result:
                    f.write("**结果**:\n")
                    f.write(f"```json\n{json.dumps(step.result, ensure_ascii=False, indent=2)}\n```\n\n")
                
                if step.error:
                    f.write(f"**错误**: {step.error}\n\n")
                
                f.write("---\n\n")
        
        return filepath
    
    def save_rag_result(
        self, 
        query: str,
        query_type: str,
        rewritten_queries: List[RewrittenQuery],
        retrieval_results: List[RetrievalResult],
        step_costs: Dict[str, Dict[str, float]],
        total_cost: float
    ) -> str:
        """保存RAG结果到JSON文件。
        
        保存完整的API检索数据，包括：
        - 原始查询和类型
        - 查询改写结果
        - 文档检索结果（doc_ids和summaries）
        - Chunk检索结果
        - Chunk决策结果
        - 费用信息
        
        Args:
            query: 原始查询
            query_type: 查询类型
            rewritten_queries: 改写后的查询列表
            retrieval_results: 检索结果列表
            step_costs: 各步骤的费用信息
            total_cost: 总费用
            
        Returns:
            保存的文件路径
        """
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_result_{timestamp}.json"
        filepath = os.path.join(self.rag_result_dir, filename)
        
        # 构建结果数据
        result_data = {
            "metadata": {
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "query": query,
                "query_type": query_type,
                "total_steps": len(self.process_steps),
                "total_time": sum(step.duration or 0 for step in self.process_steps),
                "total_cost_usd": total_cost,
                "total_rewritten_queries": len(rewritten_queries),
                "total_retrieval_results": len(retrieval_results),
                "successful_retrievals": sum(1 for r in retrieval_results if r.success)
            },
            "cost_analysis": {
                "total_cost_usd": total_cost,
                "step_costs": step_costs,
                "cost_breakdown": {
                    "intent_recognition": step_costs.get("意图识别", {}).get("total_cost_usd", 0),
                    "query_rewriting": step_costs.get("查询改写", {}).get("total_cost_usd", 0),
                    "generation": step_costs.get("生成", {}).get("total_cost_usd", 0)
                }
            },
            "process_steps": [
                {
                    "step_name": step.step_name,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "duration": step.duration,
                    "result": step.result,
                    "error": step.error,
                    "cost_info": step.result.get("cost_info", {}) if step.result else {}
                } for step in self.process_steps
            ],
            "rewritten_queries": [
                {
                    "original_query": q.original_query,
                    "rewritten_query": q.rewritten_query,
                    "intent_type": q.intent_type,
                    "reasoning": q.reasoning
                } for q in rewritten_queries
            ],
            "retrieval_results": []
        }
        
        # 详细保存每个检索结果
        for r in retrieval_results:
            retrieval_data = {
                "query": r.query,
                "success": r.success,
                "error": r.error,
                "doc_ids": r.doc_ids,
                "chunks": r.chunks,
                "decisions": r.decisions,
                "data": r.data
            }
            
            # 如果有完整的API数据，保存详细信息
            if r.data and isinstance(r.data, dict):
                # 保存文档检索结果
                if "chunk_result" in r.data:
                    chunk_result = r.data["chunk_result"]
                    retrieval_data["chunk_retrieval"] = {
                        "success": chunk_result.get("success", False),
                        "chunks": chunk_result.get("chunks", []),
                        "total_results": chunk_result.get("total_results", 0),
                        "response_data": chunk_result.get("response_data", {})
                    }
                
                # 保存Chunk检索结果
                if "decision_result" in r.data:
                    decision_result = r.data["decision_result"]
                    retrieval_data["chunk_decision"] = {
                        "success": decision_result.get("success", False),
                        "decisions": decision_result.get("decisions", []),
                        "total_results": decision_result.get("total_results", 0),
                        "response_data": decision_result.get("response_data", {})
                    }
            
            result_data["retrieval_results"].append(retrieval_data)
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"RAG结果已保存到: {filepath}")
        return filepath
    
    async def process(self, query: str) -> Dict[str, Any]:
        """执行完整的复杂检索生成流程。
        
        Args:
            query: 用户查询
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        total_cost = 0.0
        step_costs = {}
        
        try:
            logger.info(f"开始处理查询: {query}")
            
            # 检查API健康状态
            if not self.steps.check_api_health():
                return {
                    "success": False,
                    "error": "检索API不可用，请确保API服务器已启动",
                    "query": query
                }
            
            # 步骤1: 意图识别
            step1 = self._start_step("意图识别")
            intent_result = await self.steps.step1_intent_recognition(query)
            if intent_result is None:
                self._end_step(step1, error="意图识别失败")
                return {
                    "success": False,
                    "error": "意图识别失败",
                    "query": query
                }
            
            intent, cost_info = intent_result
            step_costs["意图识别"] = cost_info
            total_cost += cost_info.get("total_cost_usd", 0)
            
            self._end_step(step1, {
                "intent_type": intent.intent_type,
                "reasoning": intent.reasoning,
                "cost_info": cost_info
            })
            logger.info(f"意图识别完成: 类型={intent.intent_type}, 费用=${cost_info.get('total_cost_usd', 0):.6f}")

            # 获取文档总结 - 通过API获取
            step2_prep = self._start_step("获取文档总结")
            documents_result = await self.steps.get_documents_summary_via_api(query)
            doc_ids = documents_result.get("doc_ids", [])
            documents_summary = documents_result.get("summaries", "无文档总结信息")
            self._end_step(step2_prep, {
                "doc_ids": doc_ids,
                "documents_summary": documents_summary
            })

            logger.info("获取文档总结完成")
            logger.info(f"文档ID列表: {doc_ids}")
            logger.info(f"文档总结: {documents_summary}")

            # 步骤2: 查询改写
            step2 = self._start_step("查询改写")
            rewrite_result = await self.steps.step2_query_rewriting(query, intent, documents_summary)
            if rewrite_result is None:
                self._end_step(step2, error="查询改写失败")
                return {
                    "success": False,
                    "error": "查询改写失败",
                    "query": query
                }
            
            rewritten_queries, cost_info = rewrite_result
            step_costs["查询改写"] = cost_info
            total_cost += cost_info.get("total_cost_usd", 0)
            
            self._end_step(step2, {
                "query_count": len(rewritten_queries),
                "queries": [q.rewritten_query for q in rewritten_queries],
                "cost_info": cost_info
            })
            logger.info(f"查询改写完成: 生成了 {len(rewritten_queries)} 个改写版本, 费用=${cost_info.get('total_cost_usd', 0):.6f}")
            
            # 步骤3: 检索 - 使用KnowledgeRAG API
            step3 = self._start_step("检索")
            
            retrieval_results = await self.steps.step3_retrieval(rewritten_queries, doc_ids)
            success_count = sum(1 for r in retrieval_results if r.success)
            self._end_step(step3, {
                "total_queries": len(rewritten_queries),
                "success_count": success_count,
                "failure_count": len(rewritten_queries) - success_count
            })
            logger.info(f"检索完成: {success_count}/{len(retrieval_results)} 个查询成功")
            
            # 步骤4: 生成
            step4 = self._start_step("生成")
            generation_result = await self.steps.step4_generation(query, intent, retrieval_results, documents_result)
            if generation_result is None:
                self._end_step(step4, error="生成答案失败")
                return {
                    "success": False,
                    "error": "生成答案失败",
                    "query": query
                }
            
            final_answer, cost_info = generation_result
            step_costs["生成"] = cost_info
            total_cost += cost_info.get("total_cost_usd", 0)
            
            self._end_step(step4, {"generated_answer": final_answer, "cost_info": cost_info})
            logger.info(f"生成完成, 费用=${cost_info.get('total_cost_usd', 0):.6f}")
            
            # 保存结果
            markdown_file = self.save_markdown_result(
                query, intent, rewritten_queries, retrieval_results, final_answer, step_costs, total_cost
            )
            
            # 保存RAG结果
            rag_result_file = self.save_rag_result(
                query=query,
                query_type=f"{'信息汇总型' if intent.intent_type == 1 else '事实型'}",
                rewritten_queries=rewritten_queries,
                retrieval_results=retrieval_results,
                step_costs=step_costs,
                total_cost=total_cost
            )
            
            # 计算总处理时间
            total_time = time.time() - start_time
            
            logger.info(f"处理完成，总耗时: {total_time:.2f}秒, 总费用: ${total_cost:.6f}")
            logger.info(f"Markdown结果保存到: {markdown_file}")
            logger.info(f"RAG结果保存到: {rag_result_file}")
            
            return {
                "success": True,
                "query": query,
                "intent": {
                    "intent_type": intent.intent_type,
                    "reasoning": intent.reasoning
                },
                "rewritten_queries": len(rewritten_queries),
                "retrieval_success_rate": f"{success_count}/{len(retrieval_results)}",
                "final_answer": final_answer,
                "total_time": total_time,
                "total_cost_usd": total_cost,
                "step_costs": step_costs,
                "markdown_file": markdown_file,
                "rag_result_file": rag_result_file,
                "process_steps": [
                    {
                        "step_name": step.step_name,
                        "duration": step.duration,
                        "success": not bool(step.error),
                        "cost_info": step.result.get("cost_info", {}) if step.result else {}
                    } for step in self.process_steps
                ]
            }
            
        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "total_cost_usd": total_cost,
                "step_costs": step_costs
            }
          
async def main():
    """主函数 - 用于测试复杂检索生成流程。"""
    
    # 配置
    config = ComplexRetriConfig(
        rag_api_url="http://localhost:8001",
        knowledge_rag_api_url="http://71.178.110.3:8955",
        model_name="gpt-4.1",  # 使用更便宜的模型进行测试
        temperature=0.1,
        max_tokens=100000,
        max_queries=3  # 减少查询数量进行测试
    )
    
    # 创建工作流实例
    workflow = ComplexRetriWorkflow(config)
    
    # 测试查询
    test_queries = [
        # "What is a large language model?"
        # "Please help me sort out the changes in the model training method"
        # "Explain the differences between different training methods"
        "Please sort out all the professional terms in these papers for me"
    ]
    
    print("=" * 80)
    print("复杂检索生成系统测试")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_queries)}: {query}")
        print(f"{'='*60}")
        
        try:
            # 执行处理
            result = await workflow.process(query)
            
            if result["success"]:
                print(f"✅ 处理成功")
                print(f"📊 意图类型: {'信息汇总型' if result['intent']['intent_type'] == 1 else '事实型'}")
                print(f"🔄 改写查询数: {result['rewritten_queries']}")
                print(f"🔍 检索成功率: {result['retrieval_success_rate']}")
                print(f"⏱️ 总耗时: {result['total_time']:.2f}秒")
                print(f"💰 总费用: ${result['total_cost_usd']:.6f} USD")
                print(f"📄 Markdown文件: {result['markdown_file']}")
                print(f"📊 RAG结果文件: {result['rag_result_file']}")
                
                # 显示各步骤费用
                print("\n💰 各步骤费用:")
                for step_name, cost_info in result['step_costs'].items():
                    step_cost = cost_info.get("total_cost_usd", 0)
                    input_tokens = cost_info.get("input_tokens", 0)
                    output_tokens = cost_info.get("output_tokens", 0)
                    print(f"  - {step_name}: ${step_cost:.6f} (输入: {input_tokens} tokens, 输出: {output_tokens} tokens)")
                
                # 显示最终答案的前200个字符
                answer_preview = result['final_answer'][:200] + "..." if len(result['final_answer']) > 200 else result['final_answer']
                print(f"\n📝 答案预览: {answer_preview}")
                
                # 显示处理步骤
                print("\n📋 处理步骤:")
                for step in result['process_steps']:
                    status = "✅" if step['success'] else "❌"
                    duration = step.get('duration', 0)
                    cost_info = step.get('cost_info', {})
                    step_cost = cost_info.get("total_cost_usd", 0)
                    print(f"  {status} {step['step_name']}: {duration:.2f}秒, 费用: ${step_cost:.6f}")
                
            else:
                print(f"❌ 处理失败: {result['error']}")
                if 'total_cost_usd' in result:
                    print(f"💰 已产生费用: ${result['total_cost_usd']:.6f} USD")
                
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
        
        print(f"\n{'='*60}")
    
    print("\n🎉 所有测试完成！")


def test_single_query(query: str):
    """测试单个查询的同步版本。
    
    Args:
        query: 要测试的查询
    """
    import asyncio
    
    async def run_test():
        config = ComplexRetriConfig(
            rag_api_url="http://localhost:8001",
            knowledge_rag_api_url="http://71.178.110.3:8955",
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4000,
            max_queries=3
        )
        
        workflow = ComplexRetriWorkflow(config)
        result = await workflow.process(query)
        
        if result["success"]:
            print(f"✅ 处理成功")
            print(f"📊 意图类型: {'信息汇总型' if result['intent']['intent_type'] == 1 else '事实型'}")
            print(f"🔄 改写查询数: {result['rewritten_queries']}")
            print(f"🔍 检索成功率: {result['retrieval_success_rate']}")
            print(f"⏱️ 总耗时: {result['total_time']:.2f}秒")
            print(f"💰 总费用: ${result['total_cost_usd']:.6f} USD")
            print(f"📄 Markdown文件: {result['markdown_file']}")
            print(f"📊 RAG结果文件: {result['rag_result_file']}")
            
            # 显示各步骤费用
            print("\n💰 各步骤费用:")
            for step_name, cost_info in result['step_costs'].items():
                step_cost = cost_info.get("total_cost_usd", 0)
                input_tokens = cost_info.get("input_tokens", 0)
                output_tokens = cost_info.get("output_tokens", 0)
                print(f"  - {step_name}: ${step_cost:.6f} (输入: {input_tokens} tokens, 输出: {output_tokens} tokens)")
            
            # 显示最终答案
            print(f"\n📝 最终答案:")
            print(f"{'='*60}")
            print(result['final_answer'])
            print(f"{'='*60}")
        else:
            print(f"❌ 处理失败: {result['error']}")
            if 'total_cost_usd' in result:
                print(f"💰 已产生费用: ${result['total_cost_usd']:.6f} USD")
    
    asyncio.run(run_test())
    

if __name__ == "__main__":
    asyncio.run(main())
          