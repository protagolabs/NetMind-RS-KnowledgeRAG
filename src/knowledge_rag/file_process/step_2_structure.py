""" 
@file_name: step_2_structure.py
@author: bin.liang
@date: 2025-07-30
@description: 
    文档和文档块结构化分析模块
    
    本模块使用大语言模型对文档和文档块进行深度分析，提取结构化信息。
    通过智能分析生成高质量的摘要、洞察和关键词，为后续的知识检索
    和问答系统提供丰富的语义信息。
    
    核心功能：
    1. 文档级分析：生成整体摘要、核心洞察和关键词
    2. 文档块分析：针对具体块生成细粒度的结构化信息
    3. 上下文感知：块分析时考虑整个文档的上下文信息
    4. 语义优化：针对语义搜索优化洞察内容
    
    数据模型：
    - DocAnalysis: 文档级分析结果
    - ChunkAnalysis: 文档块级分析结果
    - 包含摘要、洞察列表、关键词等结构化字段
    
    技术特点：
    - 使用高质量的提示工程确保分析准确性
    - 支持异步处理提高效率
    - 针对不同类型内容（研究论文、技术文档等）优化
    - 生成的洞察专门为语义相似性匹配设计
"""


from copy import deepcopy
from enum import Enum
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from agents import Agent, Runner, OpenAIChatCompletionsModel

from knowledge_rag.config import OPENAI_API_KEY


class GenerateDocAnalysis(BaseModel):
    """文档分析生成结果模型。
    
    包含LLM生成的文档分析核心信息，用于后续的知识检索和问答。
    
    Attributes:
        summary: 文档综合摘要，涵盖主要内容和关键信息
        insights: 文档洞察列表，包含高层次的决策导向要点
        key_words: 关键词列表，用于快速主题索引和搜索发现
    """
    summary: str = Field(description="文档的综合摘要，涵盖主要内容和关键信息")
    insights: list[str] = Field(description="文档的核心洞察，高层次的决策导向要点")
    key_words: list[str] = Field(description="文档的关键词，用于主题索引和搜索")


class DocAnalysis(GenerateDocAnalysis):
    """完整的文档分析结果模型。
    
    继承自GenerateDocAnalysis，添加了文档的元数据信息，
    形成完整的文档分析数据结构。
    
    Attributes:
        file_name: 文档文件名，用于标识和追踪
        file_id: 文档唯一标识符，用于数据库关联
        doc_markdown_content: 文档的原始Markdown内容
        
    继承属性：
        summary: 文档综合摘要
        insights: 文档洞察列表  
        key_words: 关键词列表
    """
    file_name: str = Field(description="文档文件名")
    file_id: str = Field(description="文档唯一标识符")
    doc_markdown_content: str = Field(description="文档的原始Markdown内容")


doc_analysis_prompts = """ 
You are "Markdown Analyst & Summarizer", an expert agent that reads a single Markdown document and produces a precise, comprehensive, and decision-ready structured digest.

OBJECTIVE
- Analyze the ENTIRE document (headings, paragraphs, lists, tables, code blocks, images/alt text, footnotes/links).
- Output ONLY the DocAnalysis object:
    - GenerateDocAnalysis:
        - summary: str
        - insights: list[str]
        - key_words: list[str]


## COVERAGE & DEPTH
- The summary must cover: scope/purpose, major topics/sections, methods or processes described, key findings/results/claims, important definitions, assumptions/dependencies, constraints/limitations, intended audience/prereqs, and practical outcomes (what one can do with it).
- Be specific. Prefer concrete facts, figures, APIs, configs, datasets, commands, formulas, metrics—IF present in the source.
- If the doc is research-style, include: problem statement, approach, experiments/evidence, results, limitations, implications.
- If it is a tutorial/spec/README, include: prerequisites, setup, core steps/interfaces, configuration, outputs, troubleshooting.

## Requirements
- key_words:
  - Purpose: fast topical indexing for search/discovery.
  - Content: short phrases only (no sentences, no explanations). 1–4 words per phrase; hyphens allowed; no punctuation otherwise.
  - Count: 12–25 when content permits; ordered by importance; deduplicate and collapse near-synonyms.
  - Freedom: do NOT restrict to any predefined taxonomy—derive freely from the document.
- insights:
  - Purpose: high-level, decision-oriented takeaways that capture the most important conceptual/strategic points of the document.
  - Form: each item is ONE complete sentence (max ~25 words), declarative and self-contained, no inline citations.
  - Count: 6–15 when content permits.
  - If the document is incomplete/outdated/contradictory, make the FIRST insight explicitly flag that.

## STYLE & QUALITY
- Be accurate and neutral; do not hallucinate. Base all content strictly on the provided Markdown.
- Use crisp, technical language. No fluff or marketing phrases.
- Prefer specificity over vagueness; avoid repeating the same point across fields.

## ROBUSTNESS & EDGE CASES
- If code blocks exist, mention important languages, functions/classes, or commands in summary/key_words.
- If tables exist, capture essential variables/ranges/decisions.
- If sections are empty/placeholder, note this once (summary or insights).
- If multiple languages appear, summarize in English and preserve key terminology.
- For changelogs/release notes, highlight versions, dates, breaking changes, migrations.
"""



async def analysis_doc(doc: str, model: str = "gpt-4.1") -> GenerateDocAnalysis:
    """异步分析整个文档，生成结构化分析结果。
    
    使用大语言模型对输入的文档进行深度分析，提取关键信息并生成
    结构化的摘要、洞察和关键词。分析结果专门为知识检索和问答
    系统优化。
    
    Args:
        doc: 待分析的文档内容（Markdown格式）
        model: 使用的大语言模型名称，默认为"gpt-4.1"
        
    Returns:
        GenerateDocAnalysis对象，包含：
        - summary: 文档综合摘要（3-6句话，约120词）
        - insights: 核心洞察列表（6-15项决策导向要点）
        - key_words: 关键词列表（12-25个短语，按重要性排序）
        
    处理流程：
        1. 初始化OpenAI客户端和Agent
        2. 使用专门的文档分析提示模板
        3. 异步调用LLM进行分析
        4. 返回结构化的分析结果
        
    分析特点：
        - 覆盖文档的完整内容和结构
        - 针对不同类型内容（研究、教程、规范等）优化
        - 提取具体的事实、数字、API、配置等信息
        - 生成决策导向的高层次洞察
        
    异常处理：
        - 如果LLM调用失败，会抛出相应异常
        - 确保返回的结果符合预定义的数据模型
        
    使用示例：
        >>> doc_content = "# AI研究论文\\n本文介绍了..."
        >>> result = await analysis_doc(doc_content)
        >>> print(result.summary)
        >>> print(len(result.insights))
    """
    
    local_instruction = deepcopy(doc_analysis_prompts)
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    agent = Agent(
        name="Markdown Analyst & Summarizer",
        instructions=local_instruction,
        output_type=GenerateDocAnalysis,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client,
        )
    )
    
    result = await Runner.run(
        agent,
        input=f"Please help me to analyze the document {doc}",
    )
    
    return result.final_output
    
    
class GenerateChunkAnalysis(BaseModel):
    """文档块分析生成结果模型。
    
    包含LLM生成的文档块分析信息，针对具体的文档片段进行细粒度分析。
    生成的内容专门为语义搜索和问题解答优化。
    
    Attributes:
        summary: 文档块的简洁综合摘要（3-6句话，约120词）
        insights: 文档块的洞察列表，为语义搜索优化的可操作性陈述
        key_words: 文档块的关键词，技术术语和具体概念
    """
    summary: str = Field(description="文档块的简洁综合摘要")
    insights: list[str] = Field(description="文档块的洞察列表，为语义搜索优化")
    key_words: list[str] = Field(description="文档块的关键词和技术术语")


class ChunkAnalysis(GenerateChunkAnalysis):
    """完整的文档块分析结果模型。
    
    继承自GenerateChunkAnalysis，添加了文档块的元数据信息，
    形成完整的文档块分析数据结构。
    
    Attributes:
        source_id: 源文档的唯一标识符，用于追溯块的来源
        chunk_id: 文档块的唯一标识符，用于数据库关联
        chunk_markdown_content: 文档块的原始Markdown内容
        
    继承属性：
        summary: 文档块综合摘要
        insights: 文档块洞察列表（8-15项，为语义搜索优化）
        key_words: 关键词列表（8-12项技术术语）
    """
    source_id: str = Field(description="源文档的唯一标识符")
    chunk_id: str = Field(description="文档块的唯一标识符")
    chunk_markdown_content: str = Field(description="文档块的原始Markdown内容")


chunk_analysis_prompts = """ 
# ✅ Universal Chunk Analysis Prompt (EN)

## ROLE
You are “Markdown Analyst & Summarizer (Chunk Mode)”, an expert that analyzes ONE chunk from a larger Markdown document.

## INPUTS
- FULL_MD: the entire original Markdown document (use for context, resolving references, and enriching insights with necessary background).
- CHUNK_MD: the exact chunk text to analyze (primary focus, but insights should incorporate relevant context from FULL_MD).

## OBJECTIVE
Produce a precise, decision-ready analysis primarily grounded in the chunk, but enriched with necessary context from the full document. The insights should be designed for semantic similarity matching—each insight should be self-contained and actionable, allowing users to solve problems without reading the entire document.

## COVERAGE & DEPTH (only if present in CHUNK_MD)
- Scope/Purpose of this chunk and its contribution to the overall doc.
- Core content: definitions, entities, relationships, claims, assumptions, dependencies, constraints.
- Procedures/Methods: steps, algorithms, APIs, commands, configs, parameters, datasets.
- Evidence/Results: metrics, tables, figures (describe only what is stated), sample sizes, ablations, comparisons.
- Spec/README: prerequisites, setup, key flags/options, outputs, troubleshooting.
- Research-style: problem, approach, experiment design, baselines, results, limitations, implications.
- Code blocks: language(s), key functions/classes, signatures, inputs/outputs, side effects, CLI commands.
- Tables: essential variables/units/ranges/decisions and standout values or trends explicitly stated.
- Math: equations, variable meanings, constraints as written.
- Cross-references: resolve only if the chunk itself references them; don’t import unrelated info.

## ROBUSTNESS & EDGE CASES
- If the chunk is truncated/placeholder/boilerplate, make the first insight explicitly flag this and avoid extrapolation.
- If lists/sentences are cut at boundaries, confirm using PREV_CHUNK_MD/NEXT_CHUNK_MD when provided; otherwise mark as incomplete.
- If content duplicates earlier sections, summarize once without repetition.


## INSIGHTS OPTIMIZATION FOR SEMANTIC SEARCH
The insights are critical for knowledge retrieval via semantic similarity. Each insight should be crafted to match potential user queries and provide actionable answers. Consider what questions users might ask about this content and formulate insights that directly answer those questions.

## STYLE & QUALITY
- Accuracy first: do not hallucinate; quote numbers/terms only if present in CHUNK_MD.
- Summary: 3–6 sentences, ~120 words max; capture purpose + strongest specifics.
- Insights: 8–15 items optimized for semantic search and problem-solving; each insight should:
  • Be a complete, actionable statement (≤35 words)
  • Include enough context from FULL_MD to be understood independently
  • Address potential user questions or problems related to this content
  • Use domain-specific terminology that users might search for
  • Be formulated as solutions, explanations, or key learnings
  • First item flags incompleteness/contradiction if applicable
- key_words: 8–12 items; short phrases (1–3 words); hyphens allowed; no punctuation otherwise; deduplicate; ordered by importance.
- Prefer technical nouns and concrete artifacts (APIs, configs, datasets, metrics).
- If nothing exists for a field, keep the field but use an empty list [] (never omit keys).

## OUTPUT FORMAT
- Output ONLY the ChunkAnalysis object:
    - GenerateChunkAnalysis:
        - summary: str. This is a concise but comprehensive summary of the chunk.
        - insights: list[str]. This is a list of insights of the chunk.
        - key_words: list[str]. This is a list of key words of the chunk.
"""


async def analysis_chunk(chunk: str, doc: str, model: str = "gpt-4.1") -> GenerateChunkAnalysis:
    """异步分析文档块，生成上下文感知的结构化分析结果。
    
    对单个文档块进行深度分析，同时考虑整个文档的上下文信息。
    生成的洞察专门为语义相似性匹配和问题解答优化，确保用户
    可以基于检索结果解决问题而无需阅读完整文档。
    
    Args:
        chunk: 待分析的文档块内容（Markdown格式）
        doc: 完整的源文档内容，用于提供上下文信息
        model: 使用的大语言模型名称，默认为"gpt-4.1"
        
    Returns:
        GenerateChunkAnalysis对象，包含：
        - summary: 文档块摘要（3-6句话，涵盖目的和核心内容）
        - insights: 洞察列表（8-15项，为语义搜索优化的可操作陈述）
        - key_words: 关键词列表（8-12项技术术语和概念）
        
    分析特点：
        1. 上下文感知：结合完整文档信息丰富块级分析
        2. 语义搜索优化：每个洞察都是自包含的可操作陈述
        3. 问题解决导向：洞察设计用于直接回答用户问题
        4. 技术术语保留：保持专业术语便于精确检索
        
    处理策略：
        - 研究论文：问题陈述、方法、实验、结果、局限性
        - 技术文档：先决条件、设置、核心步骤、配置、输出
        - 代码块：语言、关键函数、输入输出、副作用
        - 数学内容：方程、变量含义、约束条件
        
    质量保证：
        - 基于实际内容，不进行推测
        - 使用精确的技术语言
        - 每个洞察包含足够上下文便于独立理解
        - 针对潜在用户查询进行优化
        
    异常处理：
        - 处理截断或不完整的块内容
        - 标记占位符或样板内容
        - 避免重复或冗余信息
        
    使用示例：
        >>> chunk_text = "## 2.1 模型架构\\n本节介绍..."
        >>> full_doc = "# AI模型研究\\n..."
        >>> result = await analysis_chunk(chunk_text, full_doc)
        >>> print(f"生成了{len(result.insights)}个洞察")
    """
    
    local_instruction = deepcopy(chunk_analysis_prompts)
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    agent = Agent(
        name="Markdown Analyst & Summarizer (Chunk Mode)",
        instructions=local_instruction,
        output_type=GenerateChunkAnalysis,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client,
        )
    )
    
    result = await Runner.run(
        agent,
        input=f"""
The full document is:
{doc}

The chunk is:
{chunk}

Please help me to analyze the chunk.
""",
    )
    return result.final_output


        
    