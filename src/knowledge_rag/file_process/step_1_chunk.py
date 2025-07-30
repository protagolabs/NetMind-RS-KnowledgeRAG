""" 
@file_name: step_1_chunk.py
@author: bin.liang
@date: 2025-07-30
@description: 
    We use this script to chunk the markdown file into smaller chunks.
"""


from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# 初步实验，我们可以用这个 markdown 格式的进行信息的提取
def chunk_markdown_file(markdown_document: str):
    
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
    
    
# 但是我认为实际上我们要用的是这个，因为我们不能保证用户上传的文件都是很好的 markdown 格式
def chunk_by_number(markdown_document: str):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([markdown_document])
    return texts


if __name__ == "__main__":
    
    with open("/home/bin.liang/Documents/02-research/NetMind-RS-KnowledgeRAG/experiments_docs/paper_set_1/Attention Is All You Need.md", "r") as f:
        markdown_document = f.read()
    md_header_splits = chunk_markdown_file(markdown_document)
    # md_header_splits = chunk_by_number(markdown_document)
    for chunk in md_header_splits:
        print(chunk)
        print("-"*100)
    
    