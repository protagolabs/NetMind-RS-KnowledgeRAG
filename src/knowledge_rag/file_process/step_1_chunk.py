""" 
@file_name: step_1_chunk.py
@author: bin.liang
@date: 2025-07-30
@description: 
    Markdown文档分块处理模块
    
    本模块提供了将Markdown文档分割成更小块的功能，支持两种分块策略：
    1. 基于Markdown标题层次的结构化分块
    2. 基于字符数量的递归分块
    
    核心功能：
    - 解析Markdown文档的标题结构
    - 按标题层次进行智能分块
    - 保持文档的逻辑结构完整性
    - 提供灵活的分块参数配置
    
    适用场景：
    - 长文档的知识库构建
    - RAG系统的文档预处理
    - 文档内容的语义分割
"""


from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def chunk_markdown_file(markdown_document: str) -> list[str]:
    """基于Markdown标题层次的结构化分块。
    
    根据Markdown文档的标题结构（#, ##, ###）进行智能分块，
    保持文档的逻辑结构完整性。每个块包含相应的标题信息，
    便于后续的语义理解和检索。
    
    Args:
        markdown_document: 输入的Markdown文档字符串
        
    Returns:
        分块后的文档列表，每个元素是一个包含标题和内容的字符串块
        
    处理逻辑：
        1. 解析文档中的标题层次（# ## ###）
        2. 按标题结构分割文档内容
        3. 为每个块添加对应的一级标题信息
        4. 返回结构化的文档块列表
        
    示例：
        >>> markdown_text = "# 第一章\\n内容1\\n## 1.1 小节\\n内容2"
        >>> chunks = chunk_markdown_file(markdown_text)
        >>> print(len(chunks))  # 输出分块数量
        
    注意：
        - 适用于结构良好的Markdown文档
        - 依赖文档的标题层次结构
        - 对于无标题或结构混乱的文档效果有限
    """
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    chunking = []
    for chunk in md_header_splits:
        get_meta_data = chunk.metadata.get("Header 1", "")
        local_chunk = f"{get_meta_data}\n{chunk.page_content}"
        chunking.append(local_chunk)
    return chunking
    
    
def chunk_by_number(markdown_document: str) -> list:
    """基于字符数量的递归文档分块。
    
    使用递归字符分割器将文档按固定大小分割，不依赖文档的结构。
    这种方法更适合处理格式不规范或无明确结构的文档。
    
    Args:
        markdown_document: 输入的Markdown文档字符串
        
    Returns:
        LangChain Document对象列表，每个对象包含分块后的内容
        
    分块参数：
        - chunk_size: 500字符 - 每个块的目标大小
        - chunk_overlap: 50字符 - 块之间的重叠部分
        - length_function: len - 使用字符长度计算
        - is_separator_regex: False - 使用固定分隔符
        
    优势：
        - 不依赖文档结构，适用性强
        - 可控制块大小，便于向量化处理
        - 支持块重叠，保持上下文连续性
        
    适用场景：
        - 格式不规范的文档
        - 需要固定大小块的场景
        - 纯文本或混合格式文档
        
    注意：
        - 可能会在句子中间分割
        - 需要根据具体需求调整参数
        - 重叠部分会增加存储开销
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([markdown_document])
    return texts



    
    