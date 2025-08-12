""" 
@file_name: generation_agent.py
@author: Yujing Wang, Bin Liang
@date: 2025-08-11
@description: 
    答案生成代理模块
    
    本模块实现了基于检索内容的智能答案生成功能。根据用户原始查询、
    意图识别结果和检索到的相关内容，生成结构化、准确、有引用的
    最终答案。
    
    核心功能：
    1. 结构化生成：根据意图类型生成不同结构的答案
    2. 引用标注：为每个关键信息点添加来源引用
    3. 内容整合：综合多个检索结果生成连贯答案
    4. 质量控制：确保答案准确性和完整性
    
    生成策略：
    - 事实型查询：提供精确、直接的答案
    - 信息汇总型查询：提供全面、结构化的总结
    
    答案特点：
    - 严格基于检索内容，不添加额外信息
    - 为每个关键信息添加引用标记
    - 使用清晰的结构化格式
    - 确保逻辑连贯和易于阅读
"""


from copy import deepcopy
from typing import Dict, Tuple
from pydantic import BaseModel, Field

from knowledge_rag.agentic_generation.rag_agents.intent_recognition_agent import QueryIntent
from knowledge_rag.agentic_generation.utils.openai_client import OpenAIClient
from knowledge_rag.config import GENERATION_MODEL_NAME

class GenerationAgent:
    """答案生成代理类。
    
    基于用户查询、意图识别结果和检索内容，生成高质量的结构化答案。
    确保答案准确性、完整性，并为每个关键信息提供来源引用。
    
    Attributes:
        openai_client: OpenAI API客户端
        prompt: 答案生成的详细提示模板
    
    工作原理：
        1. 分析用户查询和意图类型
        2. 整合所有检索到的相关内容
        3. 根据意图类型选择生成策略
        4. 生成结构化答案并添加引用标记
    
    生成原则：
        - 严格基于检索内容，不添加外部信息
        - 为每个关键信息添加引用标记
        - 根据意图类型调整答案结构
        - 确保答案逻辑清晰、易于理解
    
    引用格式：
        使用 <reference>{n}</reference> 格式标注信息来源，
        其中 n 对应检索结果的序号。
    
    示例：
        agent = GenerationAgent()
        intent = QueryIntent(query="什么是深度学习", intent_type=0, reasoning="...")
        answer, cost = await agent.generate_answer(
            original_query="什么是深度学习",
            intent=intent,
            retrieval_text="检索到的相关内容..."
        )
    """
    
    def __init__(self):
        """初始化答案生成代理。
        
        设置OpenAI客户端和详细的答案生成提示模板。提示模板包含了
        完整的生成指令，确保生成高质量的结构化答案。
        """
        self.openai_client = OpenAIClient()
        
        self.prompt = """# Structured Answer

You are a professional AI assistant. Please answer the user's question strictly based on the retrieved content below.  
**For every key statement, fact, or summary you write, you must add a reference using the special format:**  
`<reference>{{reference_order_number}}</reference>`  
where `reference_order_number` corresponds to the order of the source in the References section at the end.

**User's original question:**  
`{original_query}`

**Question type:**  
`{intent_description}`

**Question intent:**  
`{intent_reasoning}`

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
   - Ensure your writing is logical, objective, and easy to read.

---

## Output format

- Start with a brief summary statement.
- Organize the main body using markdown section headings (##, ###), bullet points, or numbered lists.
- For **every sentence, claim, or important fact**, add `<reference>{{n}}</reference>` after it.
- Do not cite specific paragraph or line positions—only use the provided order numbers for each chunk.

"""

    async def generate_answer(
        self, 
        original_query: str, 
        intent: QueryIntent, 
        retrieval_text: str
    ) -> Tuple[str, Dict[str, float]]:
        """生成最终答案。
        
        基于用户原始查询、意图识别结果和检索内容，生成结构化的
        最终答案。答案会根据意图类型采用不同的生成策略。
        
        Args:
            original_query: 用户原始查询
            intent: 意图识别结果，包含意图类型和推理过程
            retrieval_text: 检索到的相关文本内容
            
        Returns:
            包含以下内容的元组：
            - 生成的最终答案字符串
            - 费用信息字典：包含API调用的token使用和费用统计
            
        处理流程：
            1. 根据意图类型确定生成策略描述
            2. 格式化提示模板，包含所有必要信息
            3. 调用OpenAI API生成答案
            4. 返回生成的答案和费用信息
            
        生成策略：
            - 事实型查询(type=0)：生成精确、直接的答案
            - 信息汇总型查询(type=1)：生成全面、结构化的总结
            
        质量保证：
            - 严格基于检索内容，不添加外部信息
            - 为每个关键信息添加引用标记
            - 使用清晰的结构化格式
            - 确保逻辑连贯和准确性
            
        异常处理：
            如果生成结果为空，抛出ValueError异常
        """
        
        intent_description = "information aggregation" if intent.intent_type == 1 else "fact-based"
        
        local_prompt = deepcopy(self.prompt)
        local_prompt = local_prompt.format(
            original_query=original_query, 
            intent_description=intent_description, 
            intent_reasoning=intent.reasoning, 
            retrieval_text=retrieval_text
        )
        messages = [{"role": "developer", "content": local_prompt}]
            
        final_answer, cost_info = await self.openai_client.generate_text(
            model=GENERATION_MODEL_NAME,
            messages=messages,
            temperature=0.1,
        )
        
        if final_answer is None:
            raise ValueError("Generation result is None")
        
        return final_answer, cost_info
        
        
        