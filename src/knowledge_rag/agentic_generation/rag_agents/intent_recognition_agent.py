""" 
@file_name: intent_recognition_agent.py
@author: Yujing Wang, Bin Liang
@date: 2025-08-11
@description: 
    意图识别代理模块
    
    本模块实现了智能意图识别功能，能够分析用户查询并判断其意图类型。
    主要用于区分事实型查询和信息汇总型查询，为后续的查询改写和检索
    策略提供重要依据。
    
    核心功能：
    1. 意图分类：将用户查询分为事实型(0)和信息汇总型(1)两种类型
    2. 推理分析：提供意图判断的推理过程
    3. 结构化输出：返回标准化的意图识别结果
    
    意图类型说明：
    - 事实型(0)：寻求具体事实、定义、特定信息的查询
    - 信息汇总型(1)：需要总结、归纳、列举多个方面信息的查询
"""


from copy import deepcopy
from typing import Dict, Tuple
from pydantic import BaseModel, Field

from knowledge_rag.agentic_generation.utils.openai_client import OpenAIClient
from knowledge_rag.config import INTENT_RECOGNITION_MODEL_NAME


class IntentRecognitionResult(BaseModel):
    """意图识别结果模型。
    
    Attributes:
        intent_type: 意图类型 (0: 事实型, 1: 信息汇总型)
        reasoning: 推理过程
    """
    intent_type: int = Field(..., ge=0, le=1, description="意图类型：0=事实型，1=信息汇总型")
    reasoning: str = Field(..., description="推理过程")
    
class QueryIntent(BaseModel):
    """查询意图识别结果。
    
    Attributes:
        query: 原始查询
        intent_type: 意图类型 (0: 事实型, 1: 信息汇总型)
        reasoning: 推理过程
    """
    query: str
    intent_type: int
    reasoning: str
    
    
class IntentRecognitionAgent:
    """意图识别代理类。
    
    通过大语言模型分析用户查询的意图类型，区分事实型查询和信息汇总型查询。
    这种分类对于后续的查询改写策略和检索方法选择具有重要指导意义。
    
    Attributes:
        prompt: 意图识别的提示模板
        openai_client: OpenAI API客户端
    
    工作原理：
        1. 使用预定义的提示模板分析用户查询
        2. 通过大语言模型判断查询意图类型
        3. 返回结构化的意图识别结果
    
    示例：
        agent = IntentRecognitionAgent()
        result, cost = await agent.recognize_intent("什么是机器学习？")
        # result.intent_type = 0 (事实型查询)
        
        result, cost = await agent.recognize_intent("总结深度学习的主要方法")
        # result.intent_type = 1 (信息汇总型查询)
    """
    
    def __init__(self):
        """初始化意图识别代理。
        
        设置意图识别的提示模板和OpenAI客户端。提示模板基于示例学习的方式，
        能够准确识别查询是否涉及概念列举、总结等汇总类意图。
        """
        self.prompt = """You are a master of intent recognition. Please determine whether the "User Question" involves the intent of "concept listing", such as < summary type questions >, etc. If it does, output 1; otherwise, output 0.

Example:
[User Question] : Based on the inpatient medical records of Zhuque Central Hospital, summarize the diagnostic basis and differential diagnosis of Ge Moumou.
Output: 1

[User question] : {query}
Output format: 1 or 0
"""
        self.openai_client = OpenAIClient()
    
    async def recognize_intent(self, query: str) -> Tuple[QueryIntent, Dict[str, float]]:
        """识别查询意图。
        
        分析用户输入的查询，判断其是否属于信息汇总类型的意图。
        通过大语言模型的结构化输出，确保返回准确的意图分类结果。
        
        Args:
            query: 用户输入的查询字符串
            
        Returns:
            包含以下内容的元组：
            - QueryIntent对象：包含查询、意图类型和推理过程
            - 费用信息字典：包含API调用的token使用和费用统计
            
        处理流程：
            1. 格式化提示模板，插入用户查询
            2. 调用OpenAI API进行结构化响应解析
            3. 处理API响应，构建QueryIntent对象
            4. 返回意图识别结果和费用信息
            
        异常处理：
            如果API调用失败，返回默认的意图识别结果（事实型查询）
        """
        local_prompt = deepcopy(self.prompt)
        local_prompt = local_prompt.format(query=query)
        messages = [{"role": "developer", "content": local_prompt}]
        result = await self.openai_client.parse_structured_response(
            model=INTENT_RECOGNITION_MODEL_NAME,
            messages=messages,
            response_model=IntentRecognitionResult,
            temperature=1
        )
        
        if result is None:
            return QueryIntent(
                query="",
                intent_type=0,
                reasoning="",
            ), {}
        
        parsed_result, cost_info = result
        
        return QueryIntent(
            query=query,
            intent_type=parsed_result.intent_type,
            reasoning=parsed_result.reasoning
        ), cost_info
        
        