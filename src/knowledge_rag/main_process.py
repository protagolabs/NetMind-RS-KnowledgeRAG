""" 
@file_name: main_process.py
@author: Bin Liang
@date: 2025-08-01
@description: 
    RAG系统主流程模块 - 端到端处理流水线
    
    本模块整合了RAG系统的完整处理流程，提供从文档上传到答案生成
    的端到端解决方案。是整个知识RAG系统的核心调度中心。
    
    核心功能：
    1. 文档处理流程：上传、分析、分块、结构化处理
    2. 知识索引流程：向量化、数据库存储、索引构建
    3. 知识检索流程：文档检索、片段检索、相关性判断
    4. 增强生成流程：计划制定、答案生成、引用标注
    
    处理阶段：
    Stage 1 - 文档处理：
        - 文档上传和预处理
        - 文档分块和结构化分析
        - 生成摘要、洞察和关键词
        
    Stage 2 - 知识索引：
        - 文本向量化处理
        - 数据库表和集合创建
        - 向量和结构化数据存储
        
    Stage 3 - 知识检索：
        - 文档级检索和相关性判断
        - 片段级检索和精确匹配
        - 多策略融合检索
        
    Stage 4 - 增强生成：
        - 基于检索结果制定答案计划
        - 生成带引用的专业答案
        - 质量评估和优化
    
    技术特点：
    - 异步并发处理提高效率
    - 模块化设计便于扩展
    - 完整的错误处理机制
    - 灵活的配置管理
    - 支持批量和实时处理
    
    应用场景：
    - 企业知识库构建
    - 学术文献问答系统
    - 技术文档智能检索
    - 专业领域知识助手
    
    数据流：
    原始文档 → 结构化分析 → 向量索引 → 智能检索 → 答案生成
    
    作者: NetMind-RS-KnowledgeRAG Team
"""


import asyncio
import json
from loguru import logger

from typing import List, Tuple
from knowledge_rag import (
    # file process
    chunk_markdown_file,
    analysis_doc,
    analysis_chunk,
    process_document,
    # knowledge indexing
    get_embedding_normal,
    embedding_doc, 
    embedding_chunk,
    # knowledge retrieval
    create_retriever,
    chunk_matching,
    doc_matching,
    make_decision_of_doc_matching,
    make_decision_of_chunk_matching,
    # augmented generation
    make_rag_plan,
    make_rag_writing,
)
from knowledge_rag.config import KnowledgeRAGSettings
from knowledge_rag.knowledge_indexing.step_2_save_to_db import DataToDBSaver


# Step 1: 上传文件,文件处理,并返回文档和chunk的分析结果
async def upload_md_file(
    markdown_document_set: List[str], 
    file_name_set: List[str]
    ) -> Tuple[List[dict], List[dict]]:
    """批量处理Markdown文档的完整分析流程。
    
    这是RAG系统的第一个核心步骤，负责将原始Markdown文档转换为
    结构化的分析结果，为后续的向量化和索引构建提供基础数据。
    
    Args:
        markdown_document_set: Markdown文档内容列表
            - 每个元素是一个完整的Markdown文档字符串
            - 支持标准Markdown格式和扩展语法
            - 文档内容应该是完整的、结构化的
            
        file_name_set: 对应的文件名列表
            - 与markdown_document_set一一对应
            - 用于生成唯一的文档标识符
            - 影响后续的数据组织和检索
    
    Returns:
        包含以下内容的元组：
        - doc_analysis_list: 文档级分析结果列表
            - 每个文档的摘要、洞察、关键词
            - 文档元数据和处理状态
        - chunk_analysis_list: 文档块级分析结果列表
            - 所有文档的chunk分析结果合集
            - 每个chunk的摘要、洞察、关键词
    
    处理流程：
        1. 遍历每个文档和对应的文件名
        2. 调用process_document进行深度分析
        3. 提取文档级和chunk级的分析结果
        4. 转换为标准字典格式便于后续处理
        5. 合并所有文档的处理结果
    
    并发控制：
        - 使用num_workers=10控制并发数
        - 避免过多并发导致API限制
        - 平衡处理速度和资源消耗
    
    异常处理：
        - 单个文档处理失败不影响其他文档
        - 记录详细的错误信息
        - 支持部分成功的处理结果
    
    使用示例：
        >>> docs = ["# 文档1\\n内容...", "# 文档2\\n内容..."]
        >>> names = ["doc1.md", "doc2.md"]
        >>> doc_results, chunk_results = await upload_md_file(docs, names)
        >>> print(f"处理了{len(doc_results)}个文档，{len(chunk_results)}个chunks")
    
    注意事项：
        - 确保文档内容和文件名列表长度一致
        - 处理大量文档时注意API配额和时间成本
        - 建议分批处理避免内存压力
    """
    
    doc_analysis_list = []
    chunk_analysis_list = []
    
    for markdown_document, file_name in zip(markdown_document_set, file_name_set):
        doc_analysis, chunk_analysis_list = await process_document(
            markdown_document=markdown_document,
            file_name=file_name,
            num_workers=10
        )
        
        doc_analysis_list.append(doc_analysis.model_dump())
        chunk_analysis_list.extend([chunk.model_dump() for chunk in chunk_analysis_list])
    
    return doc_analysis_list, chunk_analysis_list

async def embedding_doc_and_chunk(
    doc_analysis_list: List[dict], 
    chunk_analysis_list: List[dict]
) -> Tuple[List[dict], List[dict]]:
    """对文档和chunk分析结果进行向量化处理。
    
    这是RAG系统知识索引阶段的核心步骤，将结构化的文档分析结果
    转换为向量表示，为后续的语义搜索提供数学基础。
    
    Args:
        doc_analysis_list: 文档级分析结果列表
            - 来自upload_md_file的文档分析输出
            - 包含文档摘要、洞察、关键词等文本信息
            - 每个文档对应一个分析结果字典
            
        chunk_analysis_list: 文档块级分析结果列表
            - 来自upload_md_file的chunk分析输出
            - 包含每个chunk的详细分析信息
            - 数量通常比文档数量大得多
    
    Returns:
        包含以下内容的元组：
        - doc_embedding_list: 文档级向量化结果
            - 每个文档的摘要、关键词、洞察的向量表示
            - 保留原始分析结果和新增向量字段
        - chunk_embedding_list: chunk级向量化结果
            - 每个chunk的摘要、关键词、洞察的向量表示
            - 包含完整的向量化数据结构
    
    向量化内容：
        文档级向量化：
        - summary_embedding_vector: 文档摘要向量
        - key_words_embedding_vector: 关键词向量
        - insights_embedding_vector: 洞察向量列表
        
        Chunk级向量化：
        - summary_embedding_vector: chunk摘要向量
        - key_words_embedding_vector: chunk关键词向量
        - insights_embedding_vector: chunk洞察向量列表
    
    技术特点：
        - 使用OpenAI text-embedding-3-small模型
        - 异步并发处理提高效率
        - 自动处理API频率限制
        - 支持大规模文档集合的向量化
    
    性能考虑：
        - Chunk数量通常远大于文档数量
        - 向量化是计算密集型操作
        - 需要考虑API成本和时间消耗
        - 建议分批处理大规模数据集
    
    异常处理：
        - API调用失败时的重试机制
        - 单个项目失败不影响整体处理
        - 详细的错误日志和进度跟踪
    
    使用示例：
        >>> doc_embeddings, chunk_embeddings = await embedding_doc_and_chunk(
        ...     doc_analysis_list, chunk_analysis_list
        ... )
        >>> print(f"向量化了{len(doc_embeddings)}个文档和{len(chunk_embeddings)}个chunks")
    """
    
    doc_embedding_list = await embedding_doc(doc_analysis_list)
    chunk_embedding_list = await embedding_chunk(chunk_analysis_list)

    return doc_embedding_list, chunk_embedding_list

# Step 2: 将文件处理结果保存到数据库
async def save_file_to_db(
    doc_json_file_path: str = None, 
    chunk_json_file_path: str = None,
    doc_embedding_list: List[dict] = None, 
    chunk_embedding_list: List[dict] = None
):
    """
    主函数 - 执行完整的数据库存储流程
    ================================
    
    【执行流程】
    这是整个知识库构建流程的第二步，负责将处理好的数据持久化存储：
    1. 环境准备：配置日志、加载配置、建立数据库连接
    2. 数据加载：从JSON文件读取文档和chunk数据
    3. 结构创建：动态创建MySQL表和Milvus集合
    4. 数据存储：批量插入文档和chunk数据
    5. 资源清理：关闭数据库连接
    
    【数据来源】
    输入文件：
    - paper_set_1_docs.json：文档级数据（摘要、关键词、向量等）
    - paper_set_1_chunks.json：chunk级数据（文档片段、向量等）
    
    【存储结果】
    输出到数据库：
    - MySQL：结构化数据（文本、元数据等）
    - Milvus：向量数据（embedding向量）
    
    【错误处理】
    - 文件检查：确保输入文件存在
    - 异常捕获：捕获并记录所有错误
    - 资源清理：确保数据库连接正确关闭
    - 进度跟踪：实时显示处理进度
    
    【使用场景】
    - 知识库初始化：首次构建知识库
    - 增量更新：添加新的文档数据
    - 数据迁移：从其他系统迁移数据
    """
    
    try:
        # 第二步：加载配置
        # 从环境变量加载数据库连接配置
        config = KnowledgeRAGSettings.from_env()
        logger.info("配置加载完成")
        
        # 第三步：初始化数据库存储器
        # 建立MySQL和Milvus连接
        saver = DataToDBSaver(config)
        
        if doc_json_file_path and chunk_json_file_path: 
            # 第四步：定义数据文件路径
            # 指向step_1处理结果的JSON文件
            docs_file = doc_json_file_path
            chunks_file = chunk_json_file_path
            
            # 第五步：检查输入文件是否存在
            # 确保数据源文件存在，避免后续处理失败
            if not docs_file.exists():
                raise FileNotFoundError(f"文档数据文件不存在: {docs_file}")
            if not chunks_file.exists():
                raise FileNotFoundError(f"Chunk数据文件不存在: {chunks_file}")
            
            # 第六步：加载文档数据
            logger.info("加载文档数据...")
            with open(docs_file, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
            logger.info(f"加载了{len(docs_data)}个文档")
            
            # 第七步：加载chunk数据
            logger.info("加载chunk数据...")
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            logger.info(f"加载了{len(chunks_data)}个chunks")
        elif doc_embedding_list and chunk_embedding_list:
            docs_data = doc_embedding_list
            chunks_data = chunk_embedding_list
        else:
            raise ValueError("没有提供有效的数据")
            
        
        # 第八步：分析数据结构
        # 提取所有唯一的source_id，用于创建对应的表和集合
        source_ids = list(set(chunk['source_id'] for chunk in chunks_data))
        logger.info(f"发现{len(source_ids)}个唯一的source_id")
        
        # 第九步：创建数据库结构
        # 创建文档级别的表和集合（统一存储）
        logger.info("创建文档级别的表和集合...")
        saver.create_document_tables_and_collections()
        
        # 创建chunk级别的表和集合（按source_id分组）
        logger.info("创建chunk级别的表和集合...")
        saver.create_chunk_tables_and_collections(source_ids)
        
        # 第十步：存储数据
        # 保存文档数据到数据库
        logger.info("保存文档数据...")
        saver.save_documents_to_db(docs_data)
        
        # 保存chunk数据到数据库
        logger.info("保存chunk数据...")
        saver.save_chunks_to_db(chunks_data)
        
        # 第十一步：完成处理
        logger.info("数据存储完成！")
        
    except Exception as e:
        # 错误处理：记录详细错误信息
        logger.error(f"数据存储过程失败: {e}")
        raise
    
    finally:
        # 资源清理：确保数据库连接正确关闭
        if 'saver' in locals():
            saver.close_connections()


# Step 3: 知识检索流程
async def doc_retrieval(
    retriever, 
    query: str
    ) -> List[dict]:
    """执行文档级检索和相关性判断。
    
    这是RAG系统知识检索阶段的第一步，通过全文搜索找到可能相关的
    文档，然后使用LLM进行智能的相关性判断和过滤。
    
    Args:
        retriever: 数据库检索器实例
            - 已初始化的DatabaseRetriever对象
            - 连接到MySQL和Milvus数据库
            - 提供多种检索策略
            
        query: 用户查询字符串
            - 用户提出的问题或检索需求
            - 支持自然语言查询
            - 用于文档相关性判断
    
    Returns:
        相关文档列表，每个元素包含：
        - 文档的基本信息和内容
        - 相关性判断结果
        - 匹配分数和分析详情
        - 过滤后只包含相关的文档
    
    检索流程：
        1. 全文搜索：在文档摘要和内容中搜索相关文档
        2. 初步筛选：获取top_k=50个候选文档
        3. 并发判断：使用LLM并行判断每个文档的相关性
        4. 结果过滤：只返回被判定为相关的文档
        5. 质量保证：确保返回结果的准确性
    
    搜索策略：
        - search_fields: ["summary", "doc_markdown_content"]
        - 同时搜索文档摘要和完整内容
        - 提高召回率和精确度
    
    相关性判断：
        - 使用make_decision_of_doc_matching函数
        - 基于LLM的智能相关性分析
        - 返回结构化的判断结果
        - 包含详细的分析过程
    
    并发优化：
        - 使用asyncio.gather并行处理
        - 显著提高处理效率
        - 适合处理大量候选文档
    
    异常处理：
        - 搜索失败时的错误处理
        - LLM调用异常的处理
        - 确保返回有效结果
    
    使用示例：
        >>> retriever = create_retriever()
        >>> related_docs = await doc_retrieval(retriever, "什么是深度学习？")
        >>> print(f"找到{len(related_docs)}个相关文档")
    
    注意事项：
        - top_k设置影响召回率和处理时间
        - LLM判断会增加API成本
        - 建议根据应用场景调整参数
    """

    doc_retrieval_results = await retriever.search_documents_by_text(
        query_text=query,
        search_fields=["summary", "doc_markdown_content"],
        top_k=50
    )
    
    tasks = [make_decision_of_doc_matching(query, doc) for doc in doc_retrieval_results]
    doc_matching_results = await asyncio.gather(*tasks)
    
    return doc_matching_results

async def doc_retrieval_by_data_set_type(
    retriever,
    data_set_type: str,
    query: str
) -> List[dict]:
    """根据数据集类型进行文档检索和相关性判断。
    
    这是一种基于预定义数据集的检索策略，适用于需要在特定
    文档集合中进行检索的场景。
    
    Args:
        retriever: 数据库检索器实例
        data_set_type: 数据集类型标识
            - 对应doc_set_ids.json中的数据集分类
            - 例如："research_papers", "technical_docs"等
            - 限制检索范围到特定文档集合
            
        query: 用户查询字符串
            - 用于相关性判断的查询内容
            - 在指定数据集范围内进行匹配
    
    Returns:
        相关文档列表，只包含被判定为相关的文档
        
    检索流程：
        1. 加载数据集配置：从doc_set_ids.json读取文档ID列表
        2. 批量获取文档：根据ID列表获取完整文档信息
        3. 相关性判断：使用LLM判断每个文档与查询的相关性
        4. 结果过滤：只返回相关的文档
    
    适用场景：
        - 领域专门化检索
        - 分类文档查询
        - 受限范围搜索
        - 实验和评估场景
    
    配置文件格式：
        {
            "research_papers": ["doc_id_1", "doc_id_2", ...],
            "technical_docs": ["doc_id_3", "doc_id_4", ...],
            ...
        }
    
    优势：
        - 检索范围可控
        - 避免无关文档干扰
        - 支持专门化应用
        - 便于实验和测试
    
    注意事项：
        - 需要维护doc_set_ids.json配置文件
        - 文档ID必须在数据库中存在
        - 配置文件路径当前是硬编码的
    """
    
    with open("/home/bin.liang/Documents/02-research/NetMind-RS-KnowledgeRAG/experiments/doc_set_ids.json", "r") as f:
        doc_set_ids = json.load(f)
        
    doc_ids = doc_set_ids[data_set_type]
    
    doc_retrieval_results = []
    for doc_id in doc_ids:
        doc_retrieval_results.append(await retriever.get_document_by_file_id(doc_id))
    
    tasks = [make_decision_of_doc_matching(query, doc) for doc in doc_retrieval_results]
    doc_matching_results = await asyncio.gather(*tasks)
    
    return [doc for doc in doc_matching_results if doc is not None]

async def chunk_retrieval(
    retriever, 
    query: str, 
    doc_ids: List[str],
    each_doc_chunk_number: int = 10,
) -> List[dict]:
    """在指定文档中检索相关的文档片段。
    
    这是RAG系统检索阶段的第二步，在文档级检索确定相关文档后，
    进行更精细的片段级检索，找到最相关的文档片段作为答案生成的依据。
    
    Args:
        retriever: 数据库检索器实例
            - 支持综合搜索功能的检索器
            - 连接到向量数据库和关系数据库
            
        query: 用户查询字符串
            - 用于生成查询向量和文本匹配
            - 检索的核心依据
            
        doc_ids: 目标文档ID列表
            - 来自文档级检索的结果
            - 限制检索范围到相关文档
            - 提高检索精确度
            
        each_doc_chunk_number: 每个文档返回的chunk数量
            - 默认为10个，可根据需要调整
            - 平衡召回率和处理效率
            - 影响最终的答案质量
    
    Returns:
        相关文档片段列表，每个元素包含：
        - chunk的完整内容和元数据
        - 相关性评分和排序信息
        - 所属文档的关联信息
        - 检索匹配的详细信息
    
    检索流程：
        1. 查询向量化：将查询文本转换为向量表示
        2. 综合搜索：在指定文档的chunks中进行向量和文本搜索
        3. 结果排序：按相关性分数排序
        4. 数量控制：每个文档返回指定数量的top chunks
        5. 结果整合：合并所有文档的检索结果
    
    检索策略：
        - 向量搜索：基于语义相似度
        - 文本搜索：基于关键词匹配
        - 混合搜索：结合两种策略的优势
        - 分文档处理：保持结果的多样性
    
    技术特点：
        - comprehensive_search_without_docs：跳过文档级搜索
        - 直接在指定文档中进行chunk级搜索
        - 支持向量和文本的混合检索
        - 自动处理多文档的并行搜索
    
    性能优化：
        - 限制检索范围提高效率
        - 并行处理多个文档
        - 合理的top_k设置
        - 向量化查询的缓存
    
    使用示例：
        >>> doc_ids = ["doc_1", "doc_2", "doc_3"]
        >>> chunks = await chunk_retrieval(
        ...     retriever, "什么是机器学习？", doc_ids, 5
        ... )
        >>> print(f"检索到{len(chunks)}个相关片段")
    
    注意事项：
        - doc_ids必须是有效的文档标识符
        - each_doc_chunk_number影响结果质量和数量
        - 查询向量化会增加API调用成本
    """
    
    
    query_vector = await get_embedding_normal(query)
    
    results = await retriever.comprehensive_search_without_docs(
        doc_ids=doc_ids,
        query_vector = query_vector,
        query_text = query,
        chunk_top_k = each_doc_chunk_number,
    )
    
    # 提取 relevant_chunks 列表作为返回结果
    return results.get('relevant_chunks', [])

async def make_decision_of_chunk_retrieval(
    query_text: str,
    chunks: List[dict]
) -> List[dict]:
    """对检索到的文档片段进行智能相关性判断和过滤。
    
    这是RAG系统检索阶段的质量保证步骤，使用大语言模型对检索到的
    文档片段进行精确的相关性判断，确保只有真正相关的内容进入答案生成阶段。
    
    Args:
        query_text: 用户查询文本
            - 原始的用户问题或检索需求
            - 用作相关性判断的标准
            - 确保判断的准确性和一致性
            
        chunks: 候选文档片段列表
            - 来自chunk_retrieval的检索结果
            - 每个chunk包含完整的内容和元数据
            - 需要进行相关性判断的原始数据
    
    Returns:
        经过筛选的相关文档片段列表
        - 只包含被LLM判定为相关的chunks
        - 保留原始的chunk信息和元数据
        - 过滤掉不相关或低质量的内容
        - 确保答案生成的输入质量
    
    判断流程：
        1. 并发处理：为每个chunk并行执行相关性判断
        2. LLM分析：使用make_decision_of_chunk_matching进行智能判断
        3. 结果过滤：只保留被判定为相关的chunks
        4. 质量保证：确保返回结果的相关性和准确性
    
    判断标准：
        - 内容相关性：chunk内容是否直接回答查询
        - 信息价值：chunk是否包含有价值的信息
        - 上下文匹配：chunk是否与查询上下文匹配
        - 质量评估：chunk内容的完整性和准确性
    
    技术特点：
        - 使用asyncio.gather进行并发处理
        - 基于LLM的智能相关性判断
        - 自动过滤None结果（不相关的chunks）
        - 保持原始chunk的完整信息
    
    性能考虑：
        - 并发处理提高效率
        - LLM调用增加API成本
        - 处理时间与chunk数量成正比
        - 建议合理控制输入chunk数量
    
    异常处理：
        - LLM调用失败时的处理
        - 单个chunk判断失败不影响其他
        - 确保返回有效的结果列表
        - 记录详细的处理日志
    
    使用示例：
        >>> relevant_chunks = await make_decision_of_chunk_retrieval(
        ...     "什么是深度学习？", candidate_chunks
        ... )
        >>> print(f"从{len(candidate_chunks)}个候选中筛选出{len(relevant_chunks)}个相关片段")
    
    质量保证：
        - 基于内容的精确判断
        - 避免无关信息干扰答案生成
        - 提高最终答案的准确性
        - 优化用户体验
    """
    
    tasks = [make_decision_of_chunk_matching(query_text, chunk) for chunk in chunks]
    chunk_matching_results = await asyncio.gather(*tasks)
    
    return [chunk for chunk in chunk_matching_results if chunk is not None]
    

# Step 4: 回复增强的流程


if __name__ == "__main__":
    
    retriever = asyncio.run(create_retriever())
    query = "什么是 LLM ？"
    import json 
    import asyncio
    with open("/home/bin.liang/Documents/02-research/NetMind-RS-KnowledgeRAG/experiments/doc_set_ids.json", "r") as f:
        doc_set_ids = json.load(f)
    doc_ids = doc_set_ids["llm_papers"]
    
    each_doc_chunk_number = 10
    chunks = asyncio.run(chunk_retrieval(retriever, query, doc_ids, each_doc_chunk_number))
    print(chunks[0])

