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
import httpx
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
    score_band: str
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
You are a **Fragment–Question Relevance Judge**.  
Given a **user question** and the high-level info of **one single fragment (chunk)** — its **summary** and **insights** — decide whether this fragment **contains information related to** the question.  
**Do not assume or aggregate across other fragments or the full document.**

## Objective
Return a **fragment-only containment** judgment with:
- a **score_band** (one of: `0-20`, `20-40`, `40-60`, `60-80`, `80-100`)  
- an optional **score** (integer 0–100 **inside** the chosen band, for tie-breaking)  
- a concise **analysis_detail** grounded only in this fragment (summary + insights)

## Inputs
- **User question:** `{user_question}`
- **Fragment summary:** `{chunk_summary}`
- **Fragment insights:** `{chunk_insights}`

## Decision Criteria (Containment, Fragment-Only)
Judge the fragment **related** iff it mentions or covers any **specific elements** from the question, including:
- **Named entities**: projects, models, datasets, APIs, libraries, papers, components, standards, versions.
- **Technical elements**: formulas, parameters, metrics, tasks, constraints, configurations.
- **Contextual elements**: timeframes, environments, modalities, identifiers.
**Weak but acceptable signals**: only same-domain terms without specific overlap.  
**Not related**: no identifiable overlap with the question’s specific entities/terms/constraints.

## Procedure
1. Extract key entities/terms/constraints/timeframes from the **user question**.
2. Extract salient items from the **fragment summary** and **fragment insights**.
3. Match for **overlap** (exact names, aliases, synonyms, versions, IDs, datasets, APIs, metrics, tasks).  
4. Note **scope mismatches** (different task/dataset/modality/timeframe/audience/version). Mismatches lower the band/score but do not nullify relevance if at least one specific element matches.
5. Write a concise **analysis_detail** in the **same language as the user question**, citing **1–3 concrete cues** (short quotes or faithful paraphrases) from this fragment only. **Do not invent** beyond the fragment.

## Scoring Rubric (choose a band, then pick a score inside it)
- **0–20 — No Overlap**  
  No matching entities/terms/constraints with the question; fragment is unrelated.
- **20–40 — Weak Overlap**  
  One minor match **or** only generic same-domain similarity without concrete specifics.  
  (Pick closer to 20 when purely generic; closer to 40 when one clear specific is present.)
- **40–60 — Moderate Overlap**  
  Multiple specific matches **or** one specific element with clear detail (e.g., version/metric/parameter).  
  (Closer to 60 when ≥2 solid matches or precise definitions/configs appear.)
- **60–80 — Strong Overlap**  
  Most key elements align, or the fragment clearly addresses the **same task/dataset/API** context with specifics.  
  (Use upper 70s when coverage is broad and precise with minor mismatches.)
- **80–100 — Near-Exact / Primary Topic Match**  
  The fragment is **primarily about** the same specific item/task; rich, precise coverage, minimal mismatches.  
  (Choose 90–100 when it is essentially a direct topic match.)

> **Containment rule**: Any specific overlap ⇒ at least `20–40`.  
> **Penalty rule**: Scope/timeframe/modality/version mismatches **lower** the band/score but don’t force it to `0–20` if specific overlap remains.

## Output Format (JSON-like)
- **ChunkMatchingResult**
  - **analysis_detail**: str  — brief, evidence-based justification (1–3 cues) from this fragment only, same language as the question.
  - **score_band**: str       — one of: `"0-20"`, `"20-40"`, `"40-60"`, `"60-80"`, `"80-100"`.
  - **score**: int            — *(optional)* an integer 0–100 **inside** `score_band` for ranking/tie-breaks.

## Constraints
- Fragment-only judgment; do **not** use other fragments or external knowledge.
- No chain-of-thought; provide only concise, evidence-based `analysis_detail`.
"""



async def chunk_matching(user_question: str, chunk_summary: str, chunk_insights: str) -> ChunkMatchingResultWithIsRelated:
    """
    文档片段匹配
    """
    local_prompt = deepcopy(chunk_matching_prompts)
    prompt = local_prompt.format(
        user_question=user_question,
        chunk_summary=chunk_summary,
        chunk_insights=chunk_insights
    )

    # 关键：把 httpx.AsyncClient 传给 AsyncOpenAI(http_client=hc)
    async with httpx.AsyncClient(timeout=300) as hc:
        async with AsyncOpenAI(api_key=OPENAI_API_KEY, http_client=hc) as client:
            agent = Agent(
                name="chunk_matching",
                instructions=prompt,
                model=OpenAIChatCompletionsModel(
                    model=MATCHING_MODEL,
                    openai_client=client,  # 继续复用同一个 client
                ),
                model_settings=ModelSettings(temperature=0.0),
                output_type=ChunkMatchingResult,
            )

            result = await Runner.run(
                agent,
                input="Please help me to analyze the fragment.",
            )

            return ChunkMatchingResultWithIsRelated(
                analysis_detail=result.final_output.analysis_detail,
                score=result.final_output.score,
                is_related=result.final_output.score >= 40,
            )

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


