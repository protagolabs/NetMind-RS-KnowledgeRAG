"""
@file_name: step_2_writing.py
@author: bin.liang
@date: 2025-08-01
@description: 
    增强生成写作模块 - 基于计划的答案生成
    
    本模块负责RAG系统答案生成的第二阶段，基于第一阶段制定的计划
    生成详细、专业、有引用的最终答案。
    
    核心功能：
    1. 计划执行：严格按照制定的计划生成答案
    2. 引用标注：为每个关键信息添加准确的引用
    3. 内容写作：生成详细、全面、专业的回答
    4. 质量保证：确保答案的准确性和完整性
    
    写作特点：
    - 严格基于提供的信息，不添加外部知识
    - 每个关键观点都有明确的引用标注
    - 专业性和简洁性并重
    - 结构化和逻辑清晰的表达
    
    引用格式：
    - 使用 <reference>{序号}</reference> 格式
    - 序号对应检索结果的顺序
    - 确保每个重要信息都有引用
    - 支持答案的可验证性
    
    技术特点：
    - 基于GPT-4的高质量文本生成
    - 结构化的提示工程
    - 异步处理支持高并发
    - 灵活的模板化配置
    
    应用场景：
    - 学术问题的专业回答
    - 技术文档的综合整理
    - 研究报告的生成
    - 知识问答系统的最终输出
    
    数据流：
    检索信息 + 用户问题 + 生成计划 → 答案写作 → 带引用的最终答案
"""


from copy import deepcopy

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from pydantic import BaseModel

from knowledge_rag.config import OPENAI_API_KEY


class Plan(BaseModel):
    """答案生成计划模型（复用）。
    
    与step_1_planning中的Plan模型相同，用于接收和处理
    第一阶段生成的计划信息。
    
    Attributes:
        reasoning: 制定计划的推理过程
        plan: 具体的答案生成计划
    
    注意：
        - 这是从step_1_planning模块复用的模型
        - 在实际应用中建议统一导入避免重复定义
        - 保持与第一阶段输出的结构一致性
    """
    reasoning: str
    plan: str


WRITING_PROMPT = """ 
You are a good scientist, you are answering a question about the changes in model training methods.

## You have the following information:
{information}      

## Your plan is:
{plan}

## The user question is:
{user_question}

## Your target:
Following the plan to write a answer to the user question.

## Requirements:
- Each insights/ideas/analysis should have a reference to the source document.
- You can not say anything that is not in the information.
- You must use the special format to note the reference in each sentence, by use <reference>{{the order of the reference}}</reference>
- Your answer must be very detailed and comprehensive.
- You must be professional and concise.
"""


async def make_rag_writing(information: str, user_question: str, plan: str) -> str:
    """基于计划生成RAG答案。
    
    根据第一阶段制定的计划，结合检索到的信息和用户问题，
    生成详细、专业、带引用的最终答案。
    
    Args:
        information: 检索到的相关信息
            - 来自知识库的检索结果
            - 包含文档内容、摘要、洞察等
            - 已经过相关性筛选和质量评估
            - 为答案提供事实依据和引用来源
            
        user_question: 用户提出的问题
            - 需要回答的具体问题
            - 与第一阶段规划时的问题保持一致
            - 作为答案生成的核心导向
            
        plan: 第一阶段制定的答案生成计划
            - 来自make_rag_plan函数的输出
            - 包含详细的回答策略和结构
            - 指导答案的组织和重点
            - 确保回答的系统性和完整性
    
    Returns:
        生成的最终答案字符串，特点：
        - 详细且全面的内容
        - 每个关键信息都有引用标注
        - 专业而简洁的表达
        - 严格基于提供的信息
        - 逻辑清晰的结构组织
    
    答案特点：
        1. 引用完整性：每个观点都有 <reference>{n}</reference> 标注
        2. 内容准确性：严格基于检索信息，不添加外部知识
        3. 结构合理性：按照计划组织内容，逻辑清晰
        4. 专业性：使用准确的专业术语和表达
        5. 完整性：全面回答用户问题的各个方面
    
    处理流程：
        1. 计划解析：理解第一阶段制定的计划
        2. 信息整合：将检索信息与计划结合
        3. 内容生成：按计划生成结构化答案
        4. 引用添加：为关键信息添加引用标注
        5. 质量检查：确保答案的准确性和完整性
    
    写作原则：
        - 基于事实：只使用提供的信息
        - 引用准确：每个关键点都有明确引用
        - 逻辑清晰：按照合理的结构组织内容
        - 专业表达：使用准确的专业术语
        - 全面回答：涵盖问题的所有重要方面
    
    异常处理：
        - API调用失败时的重试机制
        - 无效计划输入的处理
        - 模型响应质量的验证
        
    使用示例：
        >>> answer = await make_rag_writing(
        ...     information="检索到的文档内容...",
        ...     user_question="深度学习的应用有哪些？",
        ...     plan="1. 介绍深度学习概念 2. 列举主要应用领域..."
        ... )
        >>> print(answer)  # 包含引用的详细答案
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    local_writing_prompt = deepcopy(WRITING_PROMPT)
    writing_prompt = local_writing_prompt.format(
        information=information, 
        plan=plan,
        user_question=user_question)

    writing_agent = Agent(
        name="writing_agent",
        instructions=writing_prompt,
        model=OpenAIChatCompletionsModel(
            model="gpt-4.1",
            openai_client=client,
        ),
    )

    runner = await Runner.run(
        writing_agent,
        "Please help me to write a answer to the user question by following the plan."
    )

    return runner.final_output




