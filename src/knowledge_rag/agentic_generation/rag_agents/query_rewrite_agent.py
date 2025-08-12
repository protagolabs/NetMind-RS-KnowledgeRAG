""" 
@file_name: query_rewrite_agent.py
@author: Yujing Wang, Bin Liang
@date: 2025-08-11
@description: 
    查询改写代理模块
    
    本模块实现了智能查询改写功能，基于用户原始查询、意图识别结果和
    文档摘要信息，生成多个不同角度的改写查询，以提高检索的召回率
    和准确性。
    
    核心功能：
    1. 多角度改写：从不同角度重新表述用户查询
    2. 意图感知：基于意图类型调整改写策略
    3. 上下文感知：利用文档摘要信息指导改写
    4. 多样性保证：生成多个具有差异性的改写版本
    
    改写策略：
    - 事实型查询：生成更具体、更精确的检索表达
    - 信息汇总型查询：生成覆盖更多方面和视角的查询
    
    应用场景：
    - 提高检索召回率：通过多个查询覆盖更多相关内容
    - 增强检索精度：针对不同意图优化查询表达
    - 克服表达局限：弥补用户查询表达不准确的问题
"""


from copy import deepcopy
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple

from knowledge_rag.config import QUERY_REWRITE_MODEL_NAME
from knowledge_rag.agentic_generation.rag_agents.intent_recognition_agent import QueryIntent
from knowledge_rag.agentic_generation.utils.openai_client import OpenAIClient


class RewrittenQuery(BaseModel):
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
    
    
class QueryRewriteAgent:
    """查询改写代理类。
    
    基于用户原始查询、意图识别结果和文档摘要，生成多个改写查询。
    通过从不同角度重新表述查询，提高检索系统的召回率和精确度。
    
    Attributes:
        prompt: 查询改写的提示模板
        openai_client: OpenAI API客户端
    
    工作原理：
        1. 分析原始查询和意图类型
        2. 结合文档摘要信息作为上下文
        3. 生成多个不同角度的改写查询
        4. 确保改写查询的多样性和相关性
    
    改写策略：
        - 事实型查询：生成更精确的检索表达
        - 信息汇总型查询：生成覆盖多个方面的查询
        - 利用文档摘要：确保改写查询与知识库内容对齐
    
    示例：
        agent = QueryRewriteAgent()
        intent = QueryIntent(query="什么是深度学习", intent_type=0, reasoning="...")
        rewritten_queries, cost = await agent.rewrite_query(
            query="什么是深度学习", 
            intent=intent, 
            documents_summary="文档摘要..."
        )
    """
    
    def __init__(self):
        """初始化查询改写代理。
        
        设置查询改写的提示模板和OpenAI客户端。提示模板包含了详细的
        改写指令，能够根据不同意图类型生成合适的改写查询。
        """
        self.prompt = """You are an expert in information retrieval and query rewriting.
Given:
- The original user query: "{query}"
- The query type: {intent_description}
- The reasoning/intention behind the query: {intent_reasoning}
- A summary (or keywords) of all documents in the current database: {documents_summary}

Your task:
1. Based on the query type and intent, rewrite the original query into multiple alternative versions, each approaching the information need from a different angle.
2. For information aggregation (summarization/listing) queries, ensure the rewrites cover a wide range of relevant aspects and perspectives from the document summaries.
3. For fact-based queries, make each rewrite target specific, clearly identifiable information in the database.
4. Each rewritten query must have a clear retrieval focus and be formulated to maximize the recall of relevant information in the database, making explicit use of document summaries or keywords as reference.
5. Generate up to {max_queries} diverse rewritten queries.

Instructions:
- Use the original query and the summaries/keywords of all database documents together as context to inform your rewrites.
- Leverage your understanding of the query type and intention to tailor each rewrite for optimal information coverage and retrieval precision.
- Output each rewritten query as a separate item in a numbered list.

Please generate the rewritten queries now.
"""
        self.openai_client = OpenAIClient()
        
    async def rewrite_query(
        self, 
        query: str, 
        intent: QueryIntent, 
        documents_summary: str,
        max_queries: int = 2
    ) -> Tuple[List[RewrittenQuery], Dict[str, float]]:
        """改写用户查询。
        
        基于用户原始查询、意图识别结果和文档摘要信息，生成多个
        不同角度的改写查询，以提高后续检索的效果。
        
        Args:
            query: 用户原始查询
            intent: 意图识别结果，包含意图类型和推理过程
            documents_summary: 知识库中所有文档的摘要信息
            max_queries: 最大改写查询数量，默认为5
            
        Returns:
            包含以下内容的元组：
            - 改写查询列表：每个RewrittenQuery包含改写结果和理由
            - 费用信息字典：包含API调用的token使用和费用统计
            
        处理流程：
            1. 根据意图类型确定改写策略描述
            2. 格式化提示模板，包含所有上下文信息
            3. 调用OpenAI API生成结构化改写结果
            4. 解析API响应，构建RewrittenQuery对象列表
            5. 返回改写结果和费用信息
            
        改写原则：
            - 信息汇总型：确保改写覆盖多个相关方面和视角
            - 事实型：使改写更具体、更精确
            - 利用文档摘要：确保改写查询与知识库内容对齐
            - 保持多样性：每个改写从不同角度表述需求
            
        异常处理：
            如果API调用失败，返回空的改写查询列表
        """
        
        intent_description = "information aggregation" if intent.intent_type == 1 else "fact-based"
        
        local_prompt = deepcopy(self.prompt) 
        local_prompt = local_prompt.format(
            query=query, 
            intent_description=intent_description, 
            intent_reasoning=intent.reasoning, 
            documents_summary=documents_summary,
            max_queries=max_queries
        )
        messages = [{"role": "developer", "content": local_prompt}]
            
        result = await self.openai_client.parse_structured_response(
            model=QUERY_REWRITE_MODEL_NAME,
            messages=messages,
            response_model=QueryRewriteResult,
            temperature=0.2
        )
        
        if result is None:
            return [], {}
        
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
        
        
        