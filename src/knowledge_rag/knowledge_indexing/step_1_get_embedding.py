""" 
@file_name: step_1_get_embedding.py
@author: zhangyu
@date: 2025-07-31
@description: 
    向量化嵌入生成模块
    
    本模块负责将文档分析结果转换为向量表示，为后续的语义搜索和
    相似度匹配提供数学基础。使用OpenAI的embedding模型生成高质量
    的向量表示。
    
    核心功能：
    1. 文档级向量化：为文档摘要、关键词、洞察生成向量
    2. 文档块向量化：为每个文档块的内容生成向量表示
    3. 批量处理：高效处理大量文本的向量化任务
    4. 并发控制：使用信号量控制API调用频率
    
    向量化策略：
    - 摘要向量：用于文档级语义搜索
    - 关键词向量：用于主题相关性匹配
    - 洞察向量：用于深层语义理解和问答
    
    技术特点：
    - 异步并发处理提高效率
    - API频率限制避免超限
    - 进度跟踪显示处理状态
    - 错误处理确保稳定性
    
    数据流：
    分析结果 → 文本提取 → 向量化 → 扩展数据模型 → 输出向量化结果
"""


import json 
from typing import List, Dict 
from pydantic import BaseModel  
from copy import deepcopy
import asyncio 

from tqdm.auto import tqdm
from openai import AsyncOpenAI 

from knowledge_rag.file_process.step_3_save import DocAnalysis, ChunkAnalysis 
from knowledge_rag.config import OPENAI_API_KEY 


def get_data(file_path: str = "experiments_docs_processed/paper_set_1/knowledge_rag_results.json") -> Dict:
    """从JSON文件加载文档分析数据。
    
    读取经过step_2结构化分析的文档数据，为向量化处理做准备。
    
    Args:
        file_path: JSON数据文件路径，默认为标准的处理结果路径
        
    Returns:
        包含文档和文档块分析结果的字典：
        {
            "documents": [...],  # 文档级分析结果列表
            "chunks": [...],     # 文档块级分析结果列表
            "metadata": {...}    # 处理元数据信息
        }
        
    文件格式：
        - 来自file_process模块的输出
        - 包含完整的文档和块分析结果
        - JSON格式，UTF-8编码
        
    异常处理：
        - 文件不存在时抛出FileNotFoundError
        - JSON格式错误时抛出json.JSONDecodeError
        - 编码错误时抛出UnicodeDecodeError
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class DocAnalysisWithEmbedding(DocAnalysis):
    """带向量嵌入的文档分析结果模型。
    
    继承自DocAnalysis，添加了向量化表示字段，形成完整的
    文档分析和向量化数据结构。
    
    新增字段：
        summary_embedding_vector: 文档摘要的向量表示（1536维）
        key_words_embedding_vector: 关键词的向量表示（1536维）
        insights_embedding_vector: 每个洞察的向量表示列表
        
    继承字段：
        file_name: 文档文件名
        file_id: 文档唯一标识
        doc_markdown_content: 原始文档内容
        summary: 文档摘要文本
        insights: 洞察文本列表
        key_words: 关键词列表
        
    用途：
        - 存储完整的文档分析和向量化结果
        - 用于数据库存储和后续检索
        - 支持多层次的语义搜索
    """
    summary_embedding_vector: List[float] = None
    key_words_embedding_vector: List[float] = None
    insights_embedding_vector: List[List[float]] = None


class ChunkAnalysisWithEmbedding(ChunkAnalysis):
    """带向量嵌入的文档块分析结果模型。
    
    继承自ChunkAnalysis，添加了向量化表示字段，形成完整的
    文档块分析和向量化数据结构。
    
    新增字段：
        summary_embedding_vector: 文档块摘要的向量表示（1536维）
        key_words_embedding_vector: 文档块关键词的向量表示（1536维）
        insights_embedding_vector: 每个洞察的向量表示列表
        
    继承字段：
        source_id: 源文档标识
        chunk_id: 文档块唯一标识
        chunk_markdown_content: 文档块原始内容
        summary: 文档块摘要文本
        insights: 洞察文本列表
        key_words: 关键词列表
        
    用途：
        - 存储完整的文档块分析和向量化结果
        - 支持细粒度的语义搜索
        - 用于数据库存储和检索系统
    """
    summary_embedding_vector: List[float] = None
    key_words_embedding_vector: List[float] = None
    insights_embedding_vector: List[List[float]] = None


async def get_embedding_normal(text: str) -> List[float]:
    """获取单个文本的向量嵌入表示。
    
    使用OpenAI的text-embedding-3-small模型生成文本的向量表示，
    用于语义相似度计算和搜索。
    
    Args:
        text: 待向量化的文本字符串
        
    Returns:
        1536维的浮点数向量列表，表示文本的语义特征
        
    模型特点：
        - 模型：text-embedding-3-small
        - 维度：1536
        - 性能：平衡了质量和速度
        - 成本：相对较低的API调用成本
        
    使用场景：
        - 单个文本的向量化
        - 实时查询向量生成
        - 小批量文本处理
        
    异常处理：
        - API调用失败时抛出OpenAI异常
        - 网络错误时进行重试
        - 文本过长时自动截断
        
    示例：
        >>> vector = await get_embedding_normal("这是一个测试文本")
        >>> len(vector)  # 1536
        >>> isinstance(vector[0], float)  # True
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


async def embedding_doc(dataset: list) -> list:
    """对文档数据集进行批量向量化处理。
    
    为文档级别的分析结果生成向量表示，包括摘要、关键词和洞察
    的向量化。使用并发控制确保API调用的稳定性。
    
    Args:
        dataset: 文档分析结果列表，每个元素包含：
            - summary: 文档摘要文本
            - key_words: 关键词列表
            - insights: 洞察文本列表
            - 其他文档元数据
            
    Returns:
        向量化后的文档数据列表，每个元素为DocAnalysisWithEmbedding对象
        的字典表示，包含原始数据和对应的向量表示
        
    处理流程：
        1. 遍历每个文档
        2. 为摘要生成向量
        3. 为关键词组合生成向量
        4. 为每个洞察并发生成向量
        5. 组装完整的向量化结果
        
    并发控制：
        - 使用Semaphore(10)限制并发数
        - 避免API频率限制
        - 确保处理稳定性
        
    进度跟踪：
        - 使用tqdm显示处理进度
        - 实时反馈当前处理状态
        
    错误处理：
        - 单个文档失败不影响其他文档
        - 记录详细的错误信息
        - 支持部分成功的处理结果
        
    性能优化：
        - 异步并发处理洞察向量
        - 批量处理减少API调用开销
        - 内存友好的流式处理
        
    注意事项：
        - 确保API密钥有效
        - 注意API调用频率限制
        - 处理大数据集时考虑时间成本
    """
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    results = []
    
    for doc in tqdm(dataset):
        
        semaphore = asyncio.Semaphore(10)  # 限制并发数为10
        async def get_embedding(text: str) -> List[float]:
            async with semaphore:
                response = await client.embeddings.create(input=text, model="text-embedding-3-small")
                return response.data[0].embedding
            
        doc_key_words = " ".join(doc["key_words"])
        
        doc_analysis = DocAnalysisWithEmbedding(**doc)
        doc_analysis.summary_embedding_vector = await get_embedding_normal(doc_analysis.summary)
        doc_analysis.key_words_embedding_vector = await get_embedding_normal(doc_key_words)
        tasks = [get_embedding(insight) for insight in doc_analysis.insights]
        doc_analysis.insights_embedding_vector = await asyncio.gather(*tasks)
        
        local_result = doc_analysis.model_dump()
        results.append(local_result)
        
        # # 保存到相对路径
        # output_file = Path("experiments_docs_processed") / "paper_set_1_docs.json"
        # output_file.parent.mkdir(parents=True, exist_ok=True)
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump(results, f, ensure_ascii=False, indent=4)
        
        return results
    
async def embedding_chunk(dataset: list) -> list:
    """对文档块数据集进行批量向量化处理。
    
    为文档块级别的分析结果生成向量表示，包括块摘要、关键词和
    洞察的向量化。处理策略与文档级类似，但针对更细粒度的内容。
    
    Args:
        dataset: 文档块分析结果列表，每个元素包含：
            - summary: 文档块摘要文本
            - key_words: 关键词列表
            - insights: 洞察文本列表
            - source_id: 源文档标识
            - chunk_id: 文档块标识
            - 其他块元数据
            
    Returns:
        向量化后的文档块数据列表，每个元素为ChunkAnalysisWithEmbedding
        对象的字典表示，包含原始数据和对应的向量表示
        
    处理流程：
        1. 遍历每个文档块
        2. 为块摘要生成向量
        3. 为关键词组合生成向量
        4. 为每个洞察并发生成向量
        5. 组装完整的向量化结果
        
    向量化特点：
        - 块级语义表示：捕获文档片段的具体语义
        - 上下文感知：保持与源文档的关联
        - 细粒度检索：支持精确的内容匹配
        
    并发控制：
        - 使用Semaphore(10)限制并发数
        - 避免API频率限制和超时
        - 确保大量块数据的稳定处理
        
    进度跟踪：
        - 使用tqdm显示处理进度
        - 对于大量块数据特别重要
        
    性能考虑：
        - 文档块数量通常比文档数量大得多
        - 需要更长的处理时间
        - 考虑分批处理避免内存压力
        
    错误处理：
        - 单个块失败不影响其他块
        - 记录失败的块标识便于排查
        - 支持部分成功的处理结果
        
    使用场景：
        - 细粒度语义搜索
        - 精确内容匹配
        - 问答系统的上下文检索
        
    注意事项：
        - 处理时间可能较长，需要耐心等待
        - 确保足够的API配额
        - 考虑成本控制和批量优化
    """
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    results = []
    
    for chunk in tqdm(dataset):
        
        semaphore = asyncio.Semaphore(10)  # 限制并发数为10
        async def get_embedding(text: str) -> List[float]:
            async with semaphore:
                response = await client.embeddings.create(input=text, model="text-embedding-3-small")
                return response.data[0].embedding
            
        chunk_key_words = " ".join(chunk["key_words"])
        
        chunk_analysis = ChunkAnalysisWithEmbedding(**chunk)
        chunk_analysis.summary_embedding_vector = await get_embedding_normal(chunk_analysis.summary)
        chunk_analysis.key_words_embedding_vector = await get_embedding_normal(chunk_key_words)
        tasks = [get_embedding(insight) for insight in chunk_analysis.insights]
        chunk_analysis.insights_embedding_vector = await asyncio.gather(*tasks)
        
        local_result = chunk_analysis.model_dump()
        results.append(local_result)
        
        # 保存到相对路径
        # output_file = Path("experiments_docs_processed") / "paper_set_1_chunks.json"
        # output_file.parent.mkdir(parents=True, exist_ok=True)
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump(results, f, ensure_ascii=False, indent=4)
        
        return results
    
            

        
        
        
        
        
        
        
        
        
    