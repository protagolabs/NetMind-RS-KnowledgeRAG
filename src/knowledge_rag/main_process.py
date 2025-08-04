""" 
@file_name: main_process.py
@author: Bin Liang
@date: 2025-08-01
@description: 
    All the rag process will in this file. 
"""


from knowledge_rag import (
    # file process
    chunk_markdown_file,
    analysis_doc,
    analysis_chunk,
    # knowledge indexing
    get_embedding_normal,
    # knowledge retrieval
    create_retriever,
    chunk_matching,
    doc_matching,
    # augmented generation
    make_rag_plan,
    write_rag_answer,
)








