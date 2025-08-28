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


import tiktoken
import os 

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def count_tokens(text, model="gpt-4"):
    """
    计算文本的 token 数量
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_markdown_file(markdown_document: str) -> list[str]:
    """基于Markdown标题层次的结构化分块（带token数量合并）。
    
    根据Markdown文档的标题结构（#, ##, ###）进行智能分块，
    保持文档的逻辑结构完整性。每个块包含相应的标题信息，
    并确保每个块的token数量不少于1000个。
    
    Args:
        markdown_document: 输入的Markdown文档字符串
        
    Returns:
        分块后的文档列表，每个元素是一个包含标题和内容的字符串块
        
    处理逻辑：
        1. 解析文档中的标题层次（# ## ###）
        2. 按标题结构分割文档内容
        3. 为每个块添加完整的标题层次信息，保持原始格式
        4. 合并token数量小于1000的块，确保每个块至少有1000个tokens
        5. 返回结构化且token数量合适的文档块列表
        
    示例：
        >>> markdown_text = "# 第一章\\n内容1\\n## 1.1 小节\\n内容2"
        >>> chunks = chunk_markdown_file(markdown_text)
        >>> print(len(chunks))  # 输出分块数量
        
    注意：
        - 适用于结构良好的Markdown文档
        - 依赖文档的标题层次结构
        - 对于无标题或结构混乱的文档效果有限
        - 保持原始markdown标题格式（#, ##, ###）
        - 自动合并小块，确保每个chunk至少包含1000个tokens
        - 合并过程中保持文档的逻辑完整性
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
        # 构建完整的标题信息，保持markdown格式
        headers = []
        
        # 按层级顺序添加标题
        if chunk.metadata.get("Header 1"):
            headers.append(f"# {chunk.metadata['Header 1']}")
        if chunk.metadata.get("Header 2"):
            headers.append(f"## {chunk.metadata['Header 2']}")
        if chunk.metadata.get("Header 3"):
            headers.append(f"### {chunk.metadata['Header 3']}")
            
        # 组合标题和内容
        if headers:
            header_text = "\n".join(headers)
            local_chunk = f"{header_text}\n{chunk.page_content}"
        else:
            local_chunk = chunk.page_content
            
        chunking.append(local_chunk)
    
    # 合并小于1000 tokens的chunk
    merged_chunks = []
    i = 0
    min_tokens = 1000
    
    while i < len(chunking):
        current_chunk = chunking[i]
        current_tokens = count_tokens(current_chunk)
        
        # 如果当前chunk的tokens小于1000，尝试与后续chunk合并
        while current_tokens < min_tokens and i < len(chunking) - 1:
            i += 1
            # 合并时在中间添加一个换行符分隔
            current_chunk += "\n\n" + chunking[i]
            current_tokens = count_tokens(current_chunk)
        
        merged_chunks.append(current_chunk)
        i += 1
    
    return merged_chunks
    
    
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


def split_file(file_path: str, part_tokens: int = 30000, overlap: int = 2):
    """ 
    分块函数，将文件进行拆分，为了保证每个文档的 token 数在 part_tokens 以内，
    Args:
        file_path: 文件路径
        part_tokens: 每个分块的最大token数
        overlap: 分块的overlap数
    Returns:
        split_parts: 分块后的列表
    """
    
    with open(file_path, "r") as file:
        text = file.read()
        
    markdown_chunks = chunk_markdown_file(text) 
    
    split_parts = []
    local_split_parts = ""
    
    if len(markdown_chunks) >= 5:
        head_chunks = "\n".join(markdown_chunks[:5]) 
    else:   
        head_chunks = ""
    
    for i, chunk in enumerate(markdown_chunks):
        if count_tokens(local_split_parts ) <= part_tokens:
            local_split_parts += chunk + "\n"
        else:
            split_parts.append(local_split_parts)
            local_split_parts = ""
            if i >= overlap:
                for j in range(overlap):
                    if count_tokens(markdown_chunks[i-j]) <= part_tokens:
                        local_split_parts = markdown_chunks[i-j] + "\n" + local_split_parts
                    else:
                        break
            else:
                local_split_parts = chunk
    if count_tokens(local_split_parts) < part_tokens/2:
        split_parts[-1] += "\n" + local_split_parts
    else:
        split_parts.append(local_split_parts)

    return split_parts, head_chunks

    
if __name__ == "__main__":
    folder = "experiments_docs/apple"
    new_folder = "experiments_docs/apple_split"
    os.makedirs(new_folder, exist_ok=True)
    import os 
    file_list = os.listdir(folder)
    for file in file_list:
        file_path = os.path.join(folder, file)
        split_parts, head_chunks = split_file(file_path)
        print(len(split_parts))
        for i, part in enumerate(split_parts):
            with open(os.path.join(new_folder, f"{file}_part_{i}.md"), "w") as f:
                if i == 0:
                    part = f"<HEAD-INFORMATION>This is the first split part of the file {file}.</HEAD-INFORMATION>\n\n" + part
                else:
                    part = f"\n\n<Start of Whole file>\n\n{head_chunks}\n\n</Start of Whole file>\n...\n...\n...\n(Omitted the previous content)\n...\n...\n...\n\n<HEAD-INFORMATION>This is the {i+1}th split part of the file {file}.</HEAD-INFORMATION>\n\n" + part
                f.write(part)
    
    