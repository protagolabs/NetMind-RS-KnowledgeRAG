""" 
@file_name: main_process.py
@author: Bin Liang
@date: 2025-08-01
@description: 
    All the rag process will in this file. 
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
    ) -> Tuple[dict, List[dict]]:
    
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
):
    
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
    ) -> List[str]:
    """ 
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
) -> List[str]:
    
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
    doc_ids: list,
    each_doc_chunk_number: int = 10,
) -> List[dict]:
    """ 
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
    
    tasks = [make_decision_of_chunk_matching(query_text, chunk) for chunk in chunks]
    chunk_matching_results = await asyncio.gather(*tasks)
    
    return [chunk for chunk in chunk_matching_results if chunk is not None]
    

# Step 4: 回复增强的流程





