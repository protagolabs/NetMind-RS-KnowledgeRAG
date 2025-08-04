""" 
@file_name: step_1_get_embedding.py
@author: zhangyu
@date: 2025-07-31
@description: 
    获取embedding，并保存到本地
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


def get_data(file_path: str="experiments_docs_processed/paper_set_1/knowledge_rag_results.json") -> Dict:
    """
    获取数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class DocAnalysisWithEmbedding(DocAnalysis):
    summary_embedding_vector: List[float] = None
    key_words_embedding_vector: List[float] = None
    insights_embedding_vector: List[List[float]] = None


class ChunkAnalysisWithEmbedding(ChunkAnalysis):
    summary_embedding_vector: List[float] = None
    key_words_embedding_vector: List[float] = None
    insights_embedding_vector: List[List[float]] = None


async def get_embedding_normal(text: str) -> List[float]:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


async def main_doc():
    
    dataset = get_data()
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    results = []
    
    for doc in tqdm(dataset["documents"]):
        
        semaphore = asyncio.Semaphore(10)  # 限制并发数为10
        async def get_embedding(text: str) -> List[float]:
            async with semaphore:
                response = await client.embeddings.create(input=text, model="text-embedding-3-small")
                return response.data[0].embedding
        
        doc_analysis = DocAnalysisWithEmbedding(**doc)
        doc_analysis.summary_embedding_vector = await get_embedding_normal(doc_analysis.summary)
        tasks = [get_embedding(insight) for insight in doc_analysis.insights]
        doc_analysis.insights_embedding_vector = await asyncio.gather(*tasks)
        
        local_result = doc_analysis.model_dump()
        results.append(local_result)
        
        # 保存到相对路径
        output_file = Path("experiments_docs_processed") / "paper_set_1_docs.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    
async def main():
    
    dataset = get_data()
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    results = []
    
    for chunk in tqdm(dataset["chunks"]):
        
        semaphore = asyncio.Semaphore(10)  # 限制并发数为10
        async def get_embedding(text: str) -> List[float]:
            async with semaphore:
                response = await client.embeddings.create(input=text, model="text-embedding-3-small")
                return response.data[0].embedding
        
        chunk_analysis = ChunkAnalysisWithEmbedding(**chunk)
        chunk_analysis.summary_embedding_vector = await get_embedding_normal(chunk_analysis.summary)
        chunk_analysis.key_words_embedding_vector = await get_embedding_normal(chunk_analysis.key_words)
        tasks = [get_embedding(insight) for insight in chunk_analysis.insights]
        chunk_analysis.insights_embedding_vector = await asyncio.gather(*tasks)
        
        local_result = chunk_analysis.model_dump()
        results.append(local_result)
        
        # 保存到相对路径
        output_file = Path("experiments_docs_processed") / "paper_set_1_chunks.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    
            

        
        
        
        
        
        
        
        
        
    