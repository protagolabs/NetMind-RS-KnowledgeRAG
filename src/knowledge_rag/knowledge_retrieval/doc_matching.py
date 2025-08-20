""" 
@file_name: doc_matching.py
@author: bin.liang
@date: 2025-07-31
@description: 
    文档匹配模块 - 智能文档相关性判断
    
    本模块使用大语言模型（LLM）进行智能的文档-查询相关性判断，
    是RAG系统中的关键组件，负责过滤和筛选与用户查询相关的文档。
    
    核心功能：
    1. 文档相关性分析：基于文档摘要和洞察判断与查询的相关性
    2. 结构化输出：返回标准化的相关性判断结果
    3. 证据驱动：基于文档内容提供判断依据
    4. 二元决策：明确的相关/不相关判断
    
    技术特点：
    - 使用OpenAI GPT模型进行语义理解
    - Pydantic模型确保输出结构一致性
    - 基于包含性原则的判断逻辑
    - 支持细粒度的相关性评分
    
    应用场景：
    - 文档预筛选：在大量文档中快速找到相关文档
    - 检索优化：提高检索精确度和召回率
    - 内容过滤：避免无关文档干扰答案生成
    - 智能推荐：基于相关性推荐相关文档
    
    工作流程：
    用户查询 + 文档信息 → LLM分析 → 相关性判断 → 结构化结果
"""


from copy import deepcopy
from pydantic import BaseModel

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings

from knowledge_rag.config import OPENAI_API_KEY, MATCHING_MODEL


class DocMatchingResult(BaseModel):
    """文档匹配结果基础模型。
    
    用于存储文档与用户查询的匹配分析结果，提供详细的
    分析过程和量化评分。
    
    Attributes:
        analysis_detail: 详细的分析过程和判断依据
            - 描述为什么文档与查询相关或不相关
            - 基于文档摘要和洞察的具体证据
            - 帮助理解匹配决策的逻辑
            
        score: 相关性评分（0-100）
            - 0-30：不相关或弱相关
            - 31-70：中等相关性
            - 71-100：高度相关
            - 基于内容包含性和语义相似度
    
    用途：
        - 文档筛选决策的基础数据
        - 相关性排序的依据
        - 匹配质量的评估指标
    """
    analysis_detail: str
    score: int
    
class DocMatchingResultWithIsRelated(DocMatchingResult):
    """包含二元判断的文档匹配结果模型。
    
    继承自DocMatchingResult，添加了明确的二元相关性判断，
    便于快速筛选和过滤文档。
    
    新增字段：
        is_related: 二元相关性判断
            - True: 文档包含与查询相关的信息
            - False: 文档不包含相关信息
            - 基于包含性原则的明确判断
    
    继承字段：
        analysis_detail: 详细分析过程
        score: 相关性评分
    
    决策逻辑：
        - 通常score >= 21时，is_related = True
        - score < 21时，is_related = False
        - 具体阈值可根据应用场景调整
    
    用途：
        - 文档预筛选的直接依据
        - 布尔逻辑判断的基础
        - 二阶段检索的第一阶段结果
    """
    is_related: bool


doc_matching_prompts = """ 
## Role
You are a **Document–Question Relevance Judge**.  
Given a user question and high-level document info, decide whether the document **contains information related to** the question, and return structured results.

## Objective
Produce a binary containment decision and a calibrated score, with a brief, evidence-based analysis grounded **only** in the provided document summary and insights.

## Documents Information
- **Document summary:** `{doc_summary}`
- **Document insights:** `{doc_insights}`

## User Question
- **User question:** `{user_question}`

## Decision Criteria (Containment-Based, More Tolerant)
Judge a document **related** if it **mentions, overlaps with, or is clearly relevant to** any **specific elements** from the question, such as:
- Named entities (projects, models, datasets, APIs, libraries, papers, components, standards, versions).
- Technical terms, formulas, interfaces, configs, parameters, metrics, tasks, or constraints explicitly present in the question.
- Timeframes, environments, modalities, or identifiers directly overlapping with the question.

✅ **Weak overlap is still acceptable** — as long as at least one key term/entity overlaps, give a passing score.  
❌ Not related: no identifiable overlap with the question’s specific entities/terms/constraints.

## Procedure
1. Extract key entities/terms/constraints/timeframes from the **User question**.
2. Extract salient items from **Document summary** and **Document insights**.
3. Match for **overlap** (exact names, aliases, synonyms, versions, datasets, APIs, metrics, tasks).
4. Score the **strength of overlap** based on:
   - Specificity of match.
   - Number of overlapping elements.
   - Granularity (same task, dataset, API, or topic context).
5. Write a concise **analysis_detail** citing 1–3 concrete cues (short quotes or faithful paraphrases) from the inputs.  
   Do **not** invent content.

> Write `analysis_detail` in the **same language as the user question**.

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

## Output Format
- DocMatchingResult
    - analysis_detail: str : Brief evidence of overlap (cite 1–3 cues from the doc inputs).
    - score: int : The containment score (0–100, integers only).
"""


async def doc_matching(user_question: str, doc_summary: str, doc_insights: str) -> DocMatchingResult:
    """ 
    文档匹配
    """
    local_prompt = deepcopy(doc_matching_prompts)
    prompt = local_prompt.format(user_question=user_question, doc_summary=doc_summary, doc_insights=doc_insights)
    
    async with AsyncOpenAI(api_key=OPENAI_API_KEY) as client:
        agent = Agent(
            name="doc_matching",
            instructions=prompt,
            model=OpenAIChatCompletionsModel(
                model=MATCHING_MODEL,
                openai_client=client,
            ),
            model_settings=ModelSettings(temperature=0.0),
            output_type=DocMatchingResult,
        )

        result = await Runner.run(
            agent,
            input=f"Please help me to analyze the document.",
        )
        
        result = DocMatchingResultWithIsRelated(
            analysis_detail=result.final_output.analysis_detail,
            score=result.final_output.score,
            is_related=result.final_output.score >= 20,
        )
    
    return result

async def make_decision_of_doc_matching(
    user_question: str,
    doc_dic: dict
) -> DocMatchingResultWithIsRelated:
    
    result = await doc_matching(user_question, doc_dic["summary"], doc_dic["insights"])
    
    if result.is_related:
        return doc_dic
    else:
        return None
    
    