"""
@file_name: step_3_save.py
@author: bin.liang
@date: 2025-07-30
@description:
    文档处理和保存模块
    
    本模块实现了完整的文档处理流水线，从原始文档到结构化分析结果的
    端到端处理。包括文档分块、智能分析、ID生成和结果保存等功能。
    
    核心功能：
    1. 智能ID生成：为文档和文档块生成语义化、可追溯的唯一标识符
    2. 文档处理流水线：分块 -> 分析 -> 保存的完整流程
    3. 并发处理：支持多文档并行处理，提高效率
    4. 进度跟踪：实时显示处理进度和统计信息
    5. 错误处理：完善的异常处理和错误恢复机制
    
    主要组件：
    - IDGenerator：智能ID生成器，支持语义化和可追溯的ID
    - process_document：单文档处理函数，包含完整分析流程
    - process_folder：批量文档处理函数，支持文件夹级处理
    
    ID生成策略：
    - 文档ID：doc_{文件名前缀}_{内容哈希}_{时间戳}
    - 文档块ID：chunk_{文档前缀}_idx{索引}_{内容哈希}
    - 确保唯一性、可读性和可追溯性
    
    处理特点：
    - 异步并发处理提高效率
    - 实时进度显示和状态更新
    - 完整的错误处理和日志记录
    - 结构化的JSON输出格式
"""


import os
import json
import uuid
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm.asyncio import tqdm

from knowledge_rag.file_process.step_1_chunk import chunk_markdown_file
from knowledge_rag.file_process.step_2_structure import (
    analysis_chunk, analysis_doc, ChunkAnalysis, DocAnalysis
)


class IDGenerator:
    """智能ID生成器类。
    
    提供语义化和可追溯的ID生成功能，确保生成的ID既具有唯一性，
    又包含有意义的信息，便于系统管理和问题追踪。
    
    设计原则：
    1. 语义化：ID包含文件名、内容特征等有意义信息
    2. 可追溯：可以从ID中提取原始信息和关联关系
    3. 唯一性：通过哈希和时间戳确保全局唯一
    4. 可读性：使用清晰的格式和分隔符
    
    ID格式：
    - 文档ID：doc_{文件名前缀}_{内容哈希前8位}_{时间戳}
    - 文档块ID：chunk_{文档前缀}_idx{索引}_{内容哈希前6位}
    
    应用场景：
    - 知识库文档管理
    - 数据版本控制
    - 内容去重和验证
    - 系统审计和追踪
    """

    @staticmethod
    def generate_doc_id(file_name: str, content: str) -> str:
        """为文档生成语义化的唯一标识符。
        
        生成格式：doc_{文件名前缀}_{内容哈希前8位}_{时间戳}
        
        Args:
            file_name: 文档文件名（包含扩展名）
            content: 文档内容字符串
            
        Returns:
            格式化的文档ID字符串
            
        ID组成部分：
            1. 前缀：doc_ - 标识这是一个文档ID
            2. 文件名前缀：清理后的文件名（不含扩展名，限制20字符）
            3. 内容哈希：MD5哈希的前8位，用于内容验证和去重
            4. 时间戳：精确到秒的时间戳，确保唯一性
            
        特点：
            - 包含文件名信息，便于人工识别
            - 内容相同的文件具有相同的哈希部分
            - 时间戳确保即使文件名和内容相同也能区分
            - 字符清理确保ID符合系统命名规范
            
        示例：
            >>> id = IDGenerator.generate_doc_id("research_paper.md", "# AI研究...")
            >>> print(id)  # doc_research_paper_a1b2c3d4_20250101_120000
        """
        # 获取文件名（不含扩展名）
        file_stem = Path(file_name).stem
        # 清理文件名，只保留字母数字和下划线
        clean_name = "".join(
            c if c.isalnum() or c == '_' else '_' for c in file_stem
        )
        clean_name = clean_name[:20]  # 限制长度

        # 生成内容哈希
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]

        # 生成时间戳（精确到秒）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"doc_{clean_name}_{content_hash}_{timestamp}"

    @staticmethod
    def generate_chunk_id(doc_id: str, chunk_index: int, chunk_content: str) -> str:
        """为文档块生成可追溯的唯一标识符。
        
        生成格式：chunk_{文档前缀}_idx{索引}_{内容哈希前6位}
        
        Args:
            doc_id: 源文档的ID
            chunk_index: 文档块在文档中的索引位置（从0开始）
            chunk_content: 文档块的内容字符串
            
        Returns:
            格式化的文档块ID字符串
            
        ID组成部分：
            1. 前缀：chunk_ - 标识这是一个文档块ID
            2. 文档前缀：从文档ID中提取的前缀，建立关联关系
            3. 索引：idx{三位数索引}，标识块在文档中的位置
            4. 内容哈希：MD5哈希的前6位，用于内容验证
            
        特点：
            - 可以直接追溯到源文档
            - 包含块在文档中的精确位置信息
            - 内容哈希用于去重和完整性验证
            - 支持文档重构和版本对比
            
        示例：
            >>> chunk_id = IDGenerator.generate_chunk_id(
            ...     "doc_paper_a1b2c3d4_20250101_120000", 
            ...     0, 
            ...     "# 引言\\n本文介绍..."
            ... )
            >>> print(chunk_id)  # chunk_paper_idx000_e5f6a7
            
        应用：
            - 文档块检索和定位
            - 内容变更追踪
            - 块级数据完整性验证
        """
        # 获取文档ID的前缀部分
        doc_prefix = doc_id.split('_')[1] if '_' in doc_id else doc_id[:10]

        # 生成块内容哈希
        chunk_hash = hashlib.md5(
            chunk_content.encode('utf-8')
        ).hexdigest()[:6]

        return f"chunk_{doc_prefix}_idx{chunk_index:03d}_{chunk_hash}"

    @staticmethod
    def generate_simple_uuid() -> str:
        """生成简单的UUID。
        
        用于不需要语义化信息的场景，提供纯粹的唯一标识符。
        
        Returns:
            标准UUID4字符串
            
        使用场景：
            - 临时对象标识
            - 会话ID生成
            - 不需要可读性的唯一标识
        """
        return str(uuid.uuid4())

    @staticmethod
    def extract_doc_info_from_chunk_id(chunk_id: str) -> Dict[str, Any]:
        """从文档块ID中提取源文档信息。
        
        解析文档块ID，提取其中包含的文档关联信息和位置信息。
        
        Args:
            chunk_id: 文档块ID字符串
            
        Returns:
            包含提取信息的字典：
            - doc_prefix: 源文档前缀
            - chunk_index: 块索引（整数）
            - content_hash: 内容哈希
            
        示例：
            >>> info = IDGenerator.extract_doc_info_from_chunk_id(
            ...     "chunk_paper_idx000_e5f6a7"
            ... )
            >>> print(info)
            {'doc_prefix': 'paper', 'chunk_index': 0, 'content_hash': 'e5f6a7'}
            
        应用：
            - 块到文档的反向追踪
            - 文档重建和排序
            - 数据关系分析
        """
        parts = chunk_id.split('_')
        if len(parts) >= 4 and parts[0] == 'chunk':
            return {
                'doc_prefix': parts[1],
                'chunk_index': int(parts[2].replace('idx', '')),
                'content_hash': parts[3]
            }
        return {}


async def process_document(
    file_path: str = None, 
    file_name: str = None, 
    markdown_document: str = None, 
    num_workers: int = 5
) -> Tuple[DocAnalysis, List[ChunkAnalysis]]:
    """处理单个文档的完整流水线。
    
    执行从原始文档到结构化分析结果的完整处理流程，包括文档分析、
    分块、并行块分析等步骤。
    
    Args:
        file_path: 文档文件路径（与markdown_document二选一）
        file_name: 文档文件名，用于ID生成和标识
        markdown_document: 直接提供的文档内容（与file_path二选一）
        num_workers: 并行处理块的工作线程数，默认为5
        
    Returns:
        包含以下内容的元组：
        - DocAnalysis: 完整的文档级分析结果
        - List[ChunkAnalysis]: 所有文档块的分析结果列表
        
    处理流程：
        1. 内容获取：从文件路径读取或直接使用提供的内容
        2. ID生成：为文档生成唯一的语义化标识符
        3. 文档分析：使用LLM分析整个文档，生成摘要和洞察
        4. 文档分块：将文档分割为较小的语义块
        5. 并行块分析：使用信号量控制并发，分析所有块
        6. 结果整合：组装完整的分析结果
        
    并发控制：
        - 使用asyncio.Semaphore限制并发数
        - 避免过多并发请求导致API限制
        - 实时显示处理进度
        
    错误处理：
        - 文件读取异常处理
        - LLM API调用异常处理
        - 并发任务失败处理
        
    性能优化：
        - 异步并发处理提高效率
        - 批量处理减少API调用开销
        - 内存管理避免大文档内存溢出
        
    使用示例：
        >>> # 从文件处理
        >>> doc_analysis, chunk_analyses = await process_document(
        ...     file_path="research.md",
        ...     file_name="research.md",
        ...     num_workers=3
        ... )
        
        >>> # 直接处理内容
        >>> doc_analysis, chunk_analyses = await process_document(
        ...     file_name="content.md",
        ...     markdown_document="# 标题\\n内容...",
        ...     num_workers=5
        ... )
        
    注意事项：
        - file_path和markdown_document必须提供其中一个
        - num_workers数量应根据API限制和系统资源调整
        - 处理大文档时注意内存使用
    """

    if not markdown_document:
        # 读取文档内容
        with open(file_path, "r", encoding='utf-8') as f:
            markdown_document = f.read()
    else:
        markdown_document = markdown_document

    # 生成文档ID
    doc_id = IDGenerator.generate_doc_id(file_name, markdown_document)

    # 分析整个文档
    doc_analysis_result = await analysis_doc(markdown_document)
    doc_analysis = DocAnalysis(
        file_name=file_name,
        file_id=doc_id,
        doc_markdown_content=markdown_document,
        summary=doc_analysis_result.summary,
        insights=doc_analysis_result.insights,
        key_words=doc_analysis_result.key_words
    )

    # 分割文档为块
    chunks = chunk_markdown_file(markdown_document)

    # 并行分析所有块，使用5个worker
    semaphore = asyncio.Semaphore(num_workers)  # 限制并发数为5

    async def process_single_chunk(i: int, chunk: str) -> ChunkAnalysis:
        async with semaphore:
            print(f"Processing chunk {i+1}/{len(chunks)} for {file_name}")

            # 生成块ID
            chunk_id = IDGenerator.generate_chunk_id(doc_id, i, chunk)

            # 分析块
            chunk_result = await analysis_chunk(chunk, markdown_document)
            return ChunkAnalysis(
                source_id=doc_id,
                chunk_id=chunk_id,
                chunk_markdown_content=chunk,
                summary=chunk_result.summary,
                insights=chunk_result.insights,
                key_words=chunk_result.key_words
            )

    # 创建所有任务
    tasks = [process_single_chunk(i, chunk) for i, chunk in enumerate(chunks)]

    # 并行执行所有任务
    chunk_analyses = await asyncio.gather(*tasks)

    return doc_analysis, chunk_analyses


async def process_folder(
    folder_path: str, 
    output_file: str = "knowledge_rag_results.json"
) -> None:
    """批量处理文件夹中的所有Markdown文件。
    
    扫描指定文件夹中的所有.md文件，逐个进行完整的文档分析处理，
    并将结果保存到JSON文件中。支持实时进度显示和错误恢复。
    
    Args:
        folder_path: 包含Markdown文件的文件夹路径
        output_file: 输出JSON文件名，默认为"knowledge_rag_results.json"
        
    处理流程：
        1. 文件夹扫描：检查文件夹存在性，获取所有.md文件
        2. 初始化：创建结果数据结构和输出路径
        3. 逐文件处理：使用process_document处理每个文件
        4. 实时保存：每处理完一个文件就更新JSON文件
        5. 进度显示：使用tqdm显示处理进度和统计信息
        6. 错误处理：单个文件失败不影响整体处理
        
    输出结构：
        {
            "documents": [...],           # 所有文档的分析结果
            "chunks": [...],              # 所有文档块的分析结果
            "metadata": {
                "processed_at": "...",    # 处理时间
                "total_documents": N,     # 文档总数
                "total_chunks": M         # 文档块总数
            }
        }
        
    输出路径：
        {folder_path的父目录}/experiments_docs_processed/paper_set_1/{output_file}
        
    错误处理：
        - 文件夹不存在：输出错误信息并返回
        - 无Markdown文件：输出提示信息并返回
        - 单文件处理失败：记录错误，继续处理其他文件
        - JSON写入失败：不影响已处理的数据
        
    进度信息：
        - 实时显示当前处理的文件名
        - 显示每个文件的块数量
        - 累计显示总处理进度
        
    使用示例：
        >>> await process_folder(
        ...     folder_path="./documents",
        ...     output_file="my_analysis.json"
        ... )
        
    注意事项：
        - 确保文件夹路径正确且有读取权限
        - 输出目录会自动创建
        - 大量文件处理时注意API限制和处理时间
        - 建议定期备份中间结果
    """

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return

    file_list = os.listdir(folder_path)
    markdown_files = [f for f in file_list if f.endswith(".md")]

    if not markdown_files:
        print(f"No markdown files found in {folder_path}")
        return

    print(f"Found {len(markdown_files)} markdown files to process")

    all_results: Dict[str, Any] = {
        "documents": [],
        "chunks": [],
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "total_documents": len(markdown_files),
            "total_chunks": 0
        }
    }

    # 定义输出路径
    output_path = Path(folder_path).parent / "experiments_docs_processed" / "paper_set_1" / output_file

    # 使用 tqdm 进度条
    progress_bar = tqdm(markdown_files, desc="Processing documents")
    for file_name in progress_bar:
        progress_bar.set_description(f"Processing {file_name}")
        file_path = os.path.join(folder_path, file_name)

        try:
            doc_analysis, chunk_analyses = await process_document(
                file_path, file_name
            )

            # 添加到结果中
            all_results["documents"].append(doc_analysis.model_dump())
            all_results["chunks"].extend([
                chunk.model_dump() for chunk in chunk_analyses
            ])
            all_results["metadata"]["total_chunks"] += len(chunk_analyses)

            # 保存结果到JSON文件
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

            # 更新进度条信息
            progress_bar.set_postfix({
                'chunks': len(chunk_analyses),
                'total_chunks': all_results["metadata"]["total_chunks"]
            })

        except Exception as e:
            print(f"❌ Error processing {file_name}: {str(e)}")
            continue

    print("\n🎉 Processing completed!")
    print(f"📁 Results saved to: {output_path}")
    print(f"📊 Total documents: {all_results['metadata']['total_documents']}")
    print(f"📊 Total chunks: {all_results['metadata']['total_chunks']}")


def generate_uuid() -> str:
    """生成简单UUID的便捷函数。
    
    提供向后兼容的UUID生成功能，内部调用IDGenerator的方法。
    
    Returns:
        标准UUID4字符串
        
    用途：
        - 保持与旧代码的兼容性
        - 提供简单的UUID生成接口
        - 快速生成临时标识符
    """
    return IDGenerator.generate_simple_uuid()



