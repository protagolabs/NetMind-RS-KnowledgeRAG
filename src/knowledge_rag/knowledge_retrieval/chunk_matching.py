""" 
@file_name: chunk_matching.py
@author: Bin Liang
@date: 2025-07-31
@description: 
    文档片段匹配模块 - 细粒度内容相关性判断
    
    本模块负责在文档级筛选之后，对文档片段（chunks）进行更精细的
    相关性判断，确保检索到的内容片段与用户查询高度相关。
    
    核心功能：
    1. 片段相关性分析：基于chunk摘要和洞察判断相关性
    2. 细粒度匹配：比文档级匹配更精确的内容判断
    3. 独立判断：每个片段独立分析，不依赖其他片段
    4. 结构化输出：标准化的匹配结果格式
    
    技术特点：
    - 专注单个片段的内容分析
    - 避免跨片段的信息聚合假设
    - 基于包含性原则的精确判断
    - 支持细粒度的相关性评分
    
    应用场景：
    - 二阶段检索的第二阶段
    - 精确内容匹配
    - 答案生成的上下文筛选
    - 内容质量评估
    
    与文档匹配的区别：
    - 文档匹配：粗粒度，基于整体文档信息
    - 片段匹配：细粒度，基于具体片段内容
    - 片段匹配更精确，但计算成本更高
    
    工作流程：
    用户查询 + 片段信息 → LLM分析 → 相关性判断 → 结构化结果
"""


import json
import logging
from copy import deepcopy
from typing import List
from pydantic import BaseModel

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings

from knowledge_rag.config import OPENAI_API_KEY, MATCHING_MODEL

logger = logging.getLogger(__name__)


class ChunkMatchingResult(BaseModel):
    """文档片段匹配结果基础模型。
    
    用于存储文档片段与用户查询的匹配分析结果，提供详细的
    分析过程和量化评分。
    
    Attributes:
        analysis_detail: 详细的分析过程和判断依据
            - 描述为什么片段与查询相关或不相关
            - 基于片段摘要和洞察的具体证据
            - 专注于单个片段的内容分析
            - 不依赖其他片段或完整文档的信息
            
        score: 相关性评分（0-100）
            - 0-30：不相关或弱相关
            - 31-70：中等相关性
            - 71-100：高度相关
            - 基于片段内容的包含性和语义匹配度
    
    特点：
        - 独立性：每个片段独立分析
        - 精确性：比文档级匹配更精确
        - 专注性：只关注单个片段的内容
    
    用途：
        - 精确内容筛选的基础数据
        - 片段级排序的依据
        - 答案生成的上下文质量评估
    """
    analysis_detail: str
    score: int
    
class ChunkMatchingResultWithIsRelated(ChunkMatchingResult):
    """包含二元判断的文档片段匹配结果模型。
    
    继承自ChunkMatchingResult，添加了明确的二元相关性判断，
    便于快速筛选高质量的文档片段。
    
    新增字段：
        is_related: 二元相关性判断
            - True: 片段包含与查询直接相关的信息
            - False: 片段不包含相关信息
            - 基于片段内容的严格包含性判断
    
    继承字段：
        analysis_detail: 详细分析过程
        score: 相关性评分
    
    判断标准：
        - 比文档级判断更严格
        - 要求片段内容直接回答或涉及查询
        - 避免间接或模糊的相关性
    
    用途：
        - 高质量片段筛选
        - 答案生成的直接输入
        - 精确匹配的决策依据
        - 检索质量的最终保证
    """
    is_related: bool


chunk_matching_prompts = """ 
## Role
You are a **Fragment–Question Relevance Judge**. Given a user question and the high-level info of a **single text fragment (chunk)**, decide whether this fragment **contains information related to** the question and return structured results. **Do not assume or aggregate across other fragments or the full document.**

## Objective
Produce a binary **containment** decision and a calibrated score, with a brief, evidence-based analysis grounded **only** in the provided fragment summary and insights.

## Inputs
- **User question:** `{user_question}`
- **fragment summary:** `{chunk_summary}`
- **fragment insights:** `{chunk_insights}`

## Decision Criteria (Containment, Fragment-Only)
Judge the fragment **related** if it **mentions or covers** any **specific elements** from the question, including:
- Named entities (projects, models, datasets, APIs, libraries, papers, components, standards, versions).
- Technical terms, formulas, interfaces, configs, parameters, metrics, tasks, or constraints **explicitly present** in the question.
- Timeframes, environments, modalities, or identifiers that **directly overlap** with the question.

**Acceptable but weak signals:** generic same-domain terms without specific overlap.  
**Not related:** no identifiable overlap with the question’s specific entities/terms/constraints in this fragment.

## Procedure
1. Extract key entities/terms/constraints/timeframes from the **User question**.
2. Extract salient items from the **fragment summary** and **fragment insights**.
3. Match for **overlap** (exact names, aliases, synonyms, versions, IDs, datasets, APIs, metrics, tasks). This is about **containment**, not usefulness.
4. Note **scope mismatches** (different task/dataset/modality/timeframe/audience/version). Mismatches reduce the score but do not nullify overlap if at least one specific element matches.
5. Write a concise **analysis_detail** citing 1–3 concrete cues (short quotes or faithful paraphrases) from the fragment inputs. Do **not** invent or infer beyond this fragment.
6. Assign a **score** using the containment rubric below.

> Write `analysis_detail` in the **same language as the user question**.  
> Do not include chain-of-thought; provide only a brief, evidence-based justification.

## Scoring Rubric (0–100, integers only; fragment-only containment)
- **0–15**: No overlap. The fragment does not mention the question’s entities/terms/constraints.
- **20–25**: Very weak overlap. Same broad domain only; no specific terms matched.
- **30–45**: Weak overlap. Mentions **one** specific element from the question (entity/term/version/dataset/API/metric).
- **50–65**: Moderate overlap. Mentions **multiple** specific elements or one element with precise details (e.g., version, config, metric definition).
- **70–85**: Strong overlap. Covers **most** key elements or matches the **same task/dataset/API** context with clear specifics.
- **90–100**: Near-exact topic match. The fragment is **primarily about** the same specific item/task as the question.

## Output Format
- ChunkMatchingResult
    - analysis_detail: str  # Brief evidence of overlap (cite 1–3 cues from this fragment).
    - score: int            # 0–100 containment score, per rubric above.
"""


async def chunk_matching(user_question: str, chunk_summary: str, chunk_insights: str) -> ChunkMatchingResultWithIsRelated:
    """ 
    文档片段匹配
    """
    local_prompt = deepcopy(chunk_matching_prompts)
    prompt = local_prompt.format(user_question=user_question, chunk_summary=chunk_summary, chunk_insights=chunk_insights)
    
    async with AsyncOpenAI(api_key=OPENAI_API_KEY) as client:
        agent = Agent(
            name="chunk_matching",
            instructions=prompt,
            model=OpenAIChatCompletionsModel(
                model=MATCHING_MODEL,
                openai_client=client,
            ),
            model_settings=ModelSettings(temperature=0.0),
            output_type=ChunkMatchingResult,
        )

        result = await Runner.run(
            agent,
            input=f"Please help me to analyze the fragment.",
        )
        
        result = ChunkMatchingResultWithIsRelated(
            analysis_detail=result.final_output.analysis_detail,
            score=result.final_output.score,
            is_related=result.final_output.score >= 20,
        )
        
        return result

async def make_decision_of_chunk_matching(
    query_text: str,
    chunk: dict
) -> dict:
    
    # 记录chunk的所有字段，用于调试
    logger.info(f"Chunk字段: {list(chunk.keys())}")
    
    # 安全获取 summary 和 insights 字段，提供默认值
    summary = chunk.get("summary", "")
    insights = chunk.get("insights", "")
    
    # 检查字段是否缺失并记录警告
    if "summary" not in chunk:
        logger.warning(f"Chunk缺失'summary'字段，chunk_id: {chunk.get('chunk_id', 'unknown')}")
    if "insights" not in chunk:
        logger.warning(f"Chunk缺失'insights'字段，chunk_id: {chunk.get('chunk_id', 'unknown')}")
    
    # 记录字段内容类型和长度
    logger.info(f"Summary类型: {type(summary)}, 长度: {len(str(summary)) if summary else 0}")
    logger.info(f"Insights类型: {type(insights)}, 长度: {len(str(insights)) if insights else 0}")
    
    # 如果 summary 是字符串格式的 JSON，尝试解析
    if isinstance(summary, str) and summary.strip().startswith('{'):
        try:
            summary = json.loads(summary)
            logger.info("Summary成功解析为JSON")
        except json.JSONDecodeError:
            logger.warning("Summary JSON解析失败，保持原字符串")
    
    # 如果 insights 是字符串格式的 JSON，尝试解析
    if isinstance(insights, str) and insights.strip().startswith('{'):
        try:
            insights = json.loads(insights)
            logger.info("Insights成功解析为JSON")
        except json.JSONDecodeError:
            logger.warning("Insights JSON解析失败，保持原字符串")
    
    result = await chunk_matching(query_text, summary, insights)
    logger.info(f"Chunk匹配结果: 分数={result.score}, 是否相关={result.is_related}")
    
    if result.is_related:
        return chunk
    else:
        return None


