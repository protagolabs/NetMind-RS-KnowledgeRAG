"""
@file_name: step_3_save.py
@author: bin.liang
@date: 2025-07-30
@description:
    We use this script to save the chunk analysis results to a json file.
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
    """更合理的ID生成器，提供语义化和可追溯的ID"""

    @staticmethod
    def generate_doc_id(file_name: str, content: str) -> str:
        """
        为文档生成ID
        格式: doc_{文件名前缀}_{内容哈希前8位}_{时间戳}
        这样可以确保：
        1. 包含文件名信息，便于识别
        2. 内容相同的文件会有相同的哈希部分
        3. 时间戳确保唯一性
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
    def generate_chunk_id(doc_id: str, chunk_index: int,
                          chunk_content: str) -> str:
        """
        为文档块生成ID
        格式: chunk_{doc_id前缀}_idx{索引}_{内容哈希前6位}
        这样可以确保：
        1. 可以追溯到源文档
        2. 包含块在文档中的位置信息
        3. 内容哈希用于去重和验证
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
        """简单的UUID生成，用于不需要语义化的场景"""
        return str(uuid.uuid4())

    @staticmethod
    def extract_doc_info_from_chunk_id(chunk_id: str) -> Dict[str, Any]:
        """从chunk_id中提取文档信息"""
        parts = chunk_id.split('_')
        if len(parts) >= 4 and parts[0] == 'chunk':
            return {
                'doc_prefix': parts[1],
                'chunk_index': int(parts[2].replace('idx', '')),
                'content_hash': parts[3]
            }
        return {}


async def process_document(
    file_path: str=None, file_name: str=None, markdown_document: str = None, num_workers: int=5
) -> Tuple[DocAnalysis, List[ChunkAnalysis]]:
    """处理单个文档，返回文档分析和所有块分析结果"""

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
    folder_path: str, output_file: str = "knowledge_rag_results.json"
):
    """处理文件夹中的所有markdown文件"""

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


def generate_uuid():
    """保持向后兼容的简单UUID生成函数"""
    return IDGenerator.generate_simple_uuid()



