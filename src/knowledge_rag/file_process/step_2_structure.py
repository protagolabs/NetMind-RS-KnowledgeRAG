""" 
@file_name: step_2_structure.py
@author: bin.liang
@date: 2025-07-30
@description: 
    我们使用 LLM 对 chunk 进行结构化，从而得到一个更好的 chunk 结构化信息，并且进行一些 summary
"""


from copy import deepcopy
from enum import Enum
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from agents import Agent, Runner, OpenAIChatCompletionsModel

from knowledge_rag.config import OPENAI_API_KEY


class GenerateDocAnalysis(BaseModel):
    summary: str
    insights: list[str] = Field(description="The insights of the document")
    key_words: list[str] = Field(description="The key points of the document")


class DocAnalysis(GenerateDocAnalysis):
    file_name: str = Field(description="The name of the document")
    file_id: str = Field(description="The id of the document")
    doc_markdown_content: str = Field(description="The markdown content of the document")


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



async def analysis_doc(doc: str, model: str = "gpt-4.1"):
    
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
    summary: str = Field(description="The summary of the chunk")
    insights: list[str] = Field(description="The insights of the chunk")
    key_words: list[str] = Field(description="The key points of the chunk")


class ChunkAnalysis(GenerateChunkAnalysis):
    
    source_id: str = Field(description="The id of the source")
    chunk_id: str = Field(description="The id of the chunk")
    chunk_markdown_content: str = Field(description="The markdown content of the chunk")


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


async def analysis_chunk(chunk: str, doc: str, model: str = "gpt-4.1"):
    
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

if __name__ == "__main__":
    
    import asyncio
    with open("/home/bin.liang/Documents/02-research/NetMind-RS-KnowledgeRAG/experiments_docs/paper_set_1/Attention Is All You Need.md", "r") as f:
        markdown_document = f.read()
    # result = asyncio.run(analysis_doc(markdown_document))
    # print(result)    
    from knowledge_rag.file_process.step_1_chunk import chunk_markdown_file
    
    chunks = chunk_markdown_file(markdown_document) 
    
    the_chunk_we_test = chunks[2]
    result = asyncio.run(analysis_chunk(the_chunk_we_test, markdown_document))
    print(result)
        
    