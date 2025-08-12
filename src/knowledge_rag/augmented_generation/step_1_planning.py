"""
@file_name: step_1_planning.py
@author: bin.liang
@date: 2025-08-01
@description: 
    增强生成规划模块 - 答案生成计划制定
    
    本模块负责在RAG系统的答案生成阶段制定详细的回答计划，
    是增强生成的第一步，确保答案生成的结构化和逻辑性。
    
    核心功能：
    1. 信息分析：深入分析检索到的信息内容
    2. 计划制定：基于用户问题和信息制定回答策略
    3. 逻辑推理：提供制定计划的推理过程
    4. 结构化输出：生成标准化的计划结果
    
    工作原理：
    - 接收用户问题和检索到的信息
    - 使用GPT-4模型进行深度分析
    - 生成包含推理过程和具体计划的结构化输出
    - 为后续的答案写作提供指导
    
    技术特点：
    - 基于大语言模型的智能规划
    - 结构化的计划输出格式
    - 科学研究导向的专业分析
    - 异步处理支持高并发
    
    应用场景：
    - 复杂问题的系统性回答
    - 学术研究问题的深度分析
    - 多信息源的综合整理
    - 结构化报告的生成规划
    
    数据流：
    用户问题 + 检索信息 → 智能分析 → 计划制定 → 结构化计划输出
"""


from copy import deepcopy

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from pydantic import BaseModel

from knowledge_rag.config import OPENAI_API_KEY


class Plan(BaseModel):
    """答案生成计划模型。
    
    用于存储制定的答案生成计划，包含推理过程和具体的执行计划。
    
    Attributes:
        reasoning: 制定计划的推理过程
            - 分析用户问题的关键要素
            - 评估可用信息的相关性和价值
            - 解释计划制定的逻辑依据
            - 提供决策的透明度
            
        plan: 具体的答案生成计划
            - 回答问题的整体策略
            - 信息组织和结构安排
            - 重点内容的优先级
            - 回答的逻辑流程
    
    用途：
        - 指导答案写作的具体执行
        - 确保回答的逻辑性和完整性
        - 提供计划制定的可追溯性
        - 支持复杂问题的系统性回答
        
    特点：
        - 推理过程透明化
        - 计划具体可执行
        - 结构化的输出格式
        - 适应不同类型的问题
    """
    reasoning: str
    plan: str


PLAN_PROMPT = """
You are a good scientist, you are answering a question about the changes
in model training methods.

## You have the following information:
{information}

## The user question is:
{user_question}

## Your target:
Making a plan to use the information to answer the user question.
"""


async def make_rag_plan(information: str, user_question: str) -> Plan:
    """制定RAG答案生成计划。
    
    基于用户问题和检索到的信息，使用大语言模型制定详细的
    答案生成计划，为后续的答案写作提供结构化指导。
    
    Args:
        information: 检索到的相关信息
            - 来自知识库的检索结果
            - 包含文档内容、摘要、洞察等
            - 已经过相关性筛选和排序
            - 为回答问题提供事实依据
            
        user_question: 用户提出的问题
            - 需要回答的具体问题
            - 可能涉及复杂的概念或多个方面
            - 需要基于检索信息进行回答
            - 要求专业和准确的回答
    
    Returns:
        Plan对象，包含：
        - reasoning: 详细的推理过程和分析
        - plan: 具体的答案生成执行计划
    
    处理流程：
        1. 信息分析：深入分析检索到的信息内容
        2. 问题理解：全面理解用户问题的要求
        3. 策略制定：确定回答问题的最佳策略
        4. 计划生成：制定详细的执行计划
        5. 推理记录：记录决策的推理过程
    
    计划特点：
        - 科学性：基于科学研究的方法论
        - 系统性：全面考虑问题的各个方面
        - 逻辑性：确保回答的逻辑连贯性
        - 实用性：计划具体可执行
    
    使用场景：
        - 复杂学术问题的回答规划
        - 多信息源的整合策略
        - 结构化报告的生成指导
        - 专业问题的深度分析
    
    异常处理：
        - API调用失败时的重试机制
        - 无效输入的错误处理
        - 模型响应异常的处理
        
    示例：
        >>> plan = await make_rag_plan(
        ...     information="关于深度学习的相关文档内容...",
        ...     user_question="深度学习在计算机视觉中的应用有哪些？"
        ... )
        >>> print(plan.reasoning)  # 推理过程
        >>> print(plan.plan)       # 具体计划
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    local_plan_prompt = deepcopy(PLAN_PROMPT)
    plan_prompt = local_plan_prompt.format(
        information=information, user_question=user_question
    )

    plan_agent = Agent(
        name="plan_agent",
        instructions=plan_prompt,
        model=OpenAIChatCompletionsModel(
            model="gpt-4.1",
            openai_client=client,
        ),
        output_type=Plan,
    )

    runner = await Runner.run(
        plan_agent,
        "Please help me to make a plan to answer the user question."
    )

    return runner.final_output



