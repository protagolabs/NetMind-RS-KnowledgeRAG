"""
RAG智能代理子模块包
====================

本包包含了RAG系统的三个核心智能代理：

1. IntentRecognitionAgent - 意图识别代理
   - 分析用户查询的意图类型
   - 区分事实型查询和信息汇总型查询
   - 为后续处理提供意图指导

2. QueryRewriteAgent - 查询改写代理  
   - 基于意图和文档摘要改写查询
   - 生成多个不同角度的检索查询
   - 提高检索的召回率和精确度

3. GenerationAgent - 答案生成代理
   - 基于检索内容生成结构化答案
   - 为每个关键信息添加来源引用
   - 根据意图类型调整生成策略

这三个代理协同工作，构成了完整的RAG智能问答流水线。
"""

from .intent_recognition_agent import IntentRecognitionAgent
from .query_rewrite_agent import QueryRewriteAgent
from .generation_agent import GenerationAgent

__all__ = [
    "IntentRecognitionAgent",
    "QueryRewriteAgent", 
    "GenerationAgent"
]