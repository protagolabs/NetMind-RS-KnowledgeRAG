"""
文档处理模块包
==============

本包提供了完整的文档处理流水线，从原始Markdown文档到结构化分析结果的
端到端处理能力。包含文档分块、智能分析和结果保存等核心功能。

模块组成：
---------

step_1_chunk.py - 文档分块模块
    - chunk_markdown_file(): 基于Markdown标题层次的结构化分块
    - chunk_by_number(): 基于字符数量的递归分块
    - 支持两种分块策略，适应不同文档结构

step_2_structure.py - 文档结构化分析模块  
    - analysis_doc(): 文档级智能分析，生成摘要、洞察和关键词
    - analysis_chunk(): 文档块级分析，上下文感知的细粒度分析
    - 使用大语言模型进行深度语义理解

step_3_save.py - 文档处理和保存模块
    - IDGenerator: 智能ID生成器，语义化和可追溯的标识符
    - process_document(): 单文档完整处理流水线
    - process_folder(): 批量文档处理，支持文件夹级操作

主要特性：
---------
1. 智能分块：支持结构化和固定大小两种分块策略
2. 深度分析：使用LLM进行文档和文档块的智能分析
3. 并发处理：异步并发提高处理效率
4. 进度跟踪：实时显示处理进度和统计信息
5. 错误恢复：完善的异常处理和错误恢复机制
6. 结果持久化：结构化JSON格式保存分析结果

处理流程：
---------
原始文档 → 文档分块 → 智能分析 → 结构化保存

使用示例：
---------
>>> from knowledge_rag.file_process import process_document, chunk_markdown_file
>>> 
>>> # 处理单个文档
>>> doc_analysis, chunk_analyses = await process_document(
...     file_path="document.md",
...     file_name="document.md"
... )
>>> 
>>> # 分块处理
>>> chunks = chunk_markdown_file(markdown_content)

数据模型：
---------
- DocAnalysis: 文档级分析结果
- ChunkAnalysis: 文档块级分析结果
- 包含摘要、洞察列表、关键词等结构化字段

技术特点：
---------
- 异步处理提高效率
- 语义搜索优化的洞察生成
- 可追溯的ID生成策略
- 灵活的分块和分析参数配置
"""

# 导入核心功能
from .step_1_chunk import chunk_markdown_file, chunk_by_number
from .step_2_structure import (
    analysis_doc, 
    analysis_chunk,
    DocAnalysis,
    ChunkAnalysis,
    GenerateDocAnalysis,
    GenerateChunkAnalysis
)
from .step_3_save import (
    process_document,
    process_folder,
    IDGenerator,
    generate_uuid
)

# 定义公开接口
__all__ = [
    # 分块功能
    'chunk_markdown_file',
    'chunk_by_number',
    
    # 分析功能
    'analysis_doc',
    'analysis_chunk',
    
    # 数据模型
    'DocAnalysis', 
    'ChunkAnalysis',
    'GenerateDocAnalysis',
    'GenerateChunkAnalysis',
    
    # 处理流水线
    'process_document',
    'process_folder',
    
    # 工具类
    'IDGenerator',
    'generate_uuid'
]

# 版本信息
__version__ = '1.0.0'
__description__ = '文档处理和智能分析模块'
