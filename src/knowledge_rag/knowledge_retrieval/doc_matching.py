""" 
@file_name: doc_matching.py
@author: bin.liang
@date: 2025-07-31
@description: 
    我们使用 LLM 来判断用户的问题是否与文档相关，返回结构化信息来判断是否相关。
"""


from copy import deepcopy
from pydantic import BaseModel

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings

from knowledge_rag.config import OPENAI_API_KEY


class DocMatchingResult(BaseModel):
    """ 
    文档匹配结果
    """
    analysis_detail: str
    score: int
    
class DocMatchingResultWithIsRelated(DocMatchingResult):
    """ 
    文档匹配结果，包含是否相关
    """
    is_related: bool


doc_matching_prompts = """ 
## Role
You are a **Document–Question Relevance Judge**. Given a user question and high-level document info, decide whether the document **contains information related to** the question, and return structured results.

## Objective
Produce a binary containment decision and a calibrated score, with a brief, evidence-based analysis grounded **only** in the provided document summary and insights.

## Documents Information
- **Document summary:** `{doc_summary}`
- **Document insights:** `{doc_insights}`

## User Question
- **User question:** `{user_question}`

## Decision Criteria (Containment-Based)
Judge a document **related** if it **mentions or covers** any **specific elements** from the question, such as:
- Named entities (projects, models, datasets, APIs, libraries, papers, components, standards, versions).
- Technical terms, formulas, interfaces, configs, parameters, metrics, tasks, or constraints explicitly present in the question.
- Timeframes, environments, modalities, or identifiers directly overlapping with the question.

**Weak signals (acceptable but low score):** generic same-domain terms without specific overlap.  
**Not related:** no identifiable overlap with the question’s specific entities/terms/constraints.

## Procedure
1. Extract key entities/terms/constraints/timeframes from the **User question**.
2. Extract salient items from **Document summary** and **Document insights**.
3. Match for **overlap** (exact names, aliases, synonyms, versions, IDs, datasets, APIs, metrics, tasks).
4. Score the **strength of overlap** (specificity, number of matches, granularity). This is about **containment**, not usefulness.
5. Write a concise **analysis_detail** citing 1–3 concrete cues (short quotes or faithful paraphrases) from the inputs. Do **not** invent content.

> Write `analysis_detail` in the **same language as the user question**.  
> Do not include chain-of-thought; provide only a brief, evidence-based justification.

## Scoring Rubric (0–100, integers only)
- **0–15**: No overlap. The doc does not mention the question’s entities/terms/constraints.
- **20–25**: Very weak overlap. Broad domain only; no specific terms matched.
- **30–45**: Weak overlap. Mentions **one** specific element from the question (entity/term/version/dataset/API/metric).
- **50–65**: Moderate overlap. Mentions **multiple** specific elements or one element with precise details (e.g., version, config, metric definition).
- **70–85**: Strong overlap. Covers **most** key elements or matches the **same task/dataset/API** context with clear specifics.
- **90–100**: Near-exact topic match. The doc is **primarily about** the same specific item/task as the question.

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
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    agent = Agent(
        name="doc_matching",
        instructions=prompt,
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini",
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


    
    