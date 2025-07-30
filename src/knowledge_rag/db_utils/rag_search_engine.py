"""
RAGSearchEngine - RAG系统的两级搜索引擎

这个模块实现RAG系统的两级搜索逻辑：
1. 文档级搜索：在documents表和documents_vectors集合中搜索相关文档
2. Chunk级搜索：在特定文档的chunk级数据库中搜索相关chunks
3. 并行搜索：同时在多个文档的chunk级数据库中搜索

搜索流程：
1. 第一步：文档级搜索 -> 找到相关文档ID列表
2. 第二步：针对这些文档，并行进行chunk级搜索 -> 获取相关chunks

作者: XYZ-Algorithm-Team
"""

import json
import logging
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from .rag_document_manager import RAGDocumentManager
from .database_clients import MySQLClient, MilvusClient

logger = logging.getLogger(__name__)

class RAGSearchEngine:
    """
    RAG搜索引擎 - 实现两级搜索逻辑
    
    使用示例：
        # 创建搜索引擎
        search_engine = RAGSearchEngine()
        
        # 第一步：文档级搜索
        relevant_docs = search_engine.search_documents(
            keywords="机器学习 深度学习",
            query_vector=[0.1, 0.2, ...],
            vector_field="document_embedding",
            top_k=10
        )
        
        # 第二步：chunk级搜索
        doc_ids = [doc['id'] for doc in relevant_docs]
        chunk_results = search_engine.search_chunks_parallel(
            doc_ids=doc_ids,
            keywords="transformer attention",
            query_vector=[0.3, 0.4, ...],
            top_k_per_doc=5
        )
    """
    
    def __init__(self, experiment_name: str = None):
        """
        初始化RAG搜索引擎
        
        Args:
            experiment_name: 实验名称
        """
        self.rag_manager = RAGDocumentManager(experiment_name)
        self.mysql_client = self.rag_manager.mysql_client
        self.milvus_client = self.rag_manager.milvus_client
        self.db_name = self.rag_manager.db_name
        
        # 文档级向量集合名称
        self.documents_collection_name = f"{self.db_name}_documents_vectors"
        
        logger.info("RAGSearchEngine 初始化完成")
    
    # ======================================
    # 文档级搜索（第一级搜索）
    # ======================================
    
    def search_documents(
        self,
        keywords: str = None,
        query_vector: List[float] = None,
        vector_field: str = "document_embedding",
        mysql_search_fields: List[str] = None,
        similarity_threshold: float = 0.5,
        top_k: int = 20,
        combine_results: bool = True,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[Dict]:
        """
        文档级搜索 - 第一级搜索
        
        Args:
            keywords: 搜索关键词
            query_vector: 查询向量
            vector_field: 向量字段名 ("document_embedding", "summary_embedding", "keywords_embedding")
            mysql_search_fields: MySQL搜索字段，默认为 ["title", "summary", "keywords"]
            similarity_threshold: 向量相似度阈值
            top_k: 返回文档数量
            combine_results: 是否结合关键词和向量搜索结果
            keyword_weight: 关键词搜索权重
            vector_weight: 向量搜索权重
        
        Returns:
            List[Dict]: 相关文档列表，每个文档包含完整的MySQL记录
        """
        
        if not keywords and not query_vector:
            raise ValueError("必须提供关键词或查询向量中的至少一个")
        
        # 设置默认搜索字段
        if mysql_search_fields is None:
            mysql_search_fields = ["title", "summary", "keywords"]
        
        # 如果只有一种搜索条件，直接搜索
        if not combine_results or (not keywords or not query_vector):
            if keywords:
                return self.search_documents_by_keywords(keywords, mysql_search_fields, top_k)
            else:
                return self.search_documents_by_vector(query_vector, vector_field, similarity_threshold, top_k)
        
        # 混合搜索：结合关键词和向量搜索
        return self._hybrid_document_search(
            keywords, query_vector, vector_field, mysql_search_fields,
            similarity_threshold, top_k, keyword_weight, vector_weight
        )
    
    def search_documents_by_keywords(
        self, 
        keywords: str, 
        search_fields: List[str] = None,
        additional_filters: Dict = None,
        top_k: int = 20
    ) -> List[Dict]:
        """
        基于关键词搜索文档
        
        Args:
            keywords: 搜索关键词
            search_fields: 搜索字段列表
            additional_filters: 额外过滤条件
            top_k: 返回文档数量
        
        Returns:
            List[Dict]: 匹配的文档列表
        """
        
        if search_fields is None:
            search_fields = ["title", "summary", "keywords"]
        
        # 构建搜索SQL
        where_conditions = []
        params = []
        
        # 关键词搜索条件
        if keywords:
            keyword_conditions = []
            for field in search_fields:
                if field in ["summary", "keywords"]:
                    # 使用全文索引搜索
                    keyword_conditions.append(f"MATCH(`{field}`) AGAINST(%s IN NATURAL LANGUAGE MODE)")
                else:
                    # 使用LIKE搜索
                    keyword_conditions.append(f"`{field}` LIKE %s")
                    
                params.append(f"%{keywords}%" if field not in ["summary", "keywords"] else keywords)
            
            if keyword_conditions:
                where_conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        # 额外过滤条件
        if additional_filters:
            for field, value in additional_filters.items():
                where_conditions.append(f"`{field}` = %s")
                params.append(value)
        
        # 构建完整SQL
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        query_sql = f"""
        SELECT * FROM `{self.db_name}`.`documents`
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT %s
        """
        params.append(top_k)
        
        try:
            results = self.mysql_client.fetch_all(query_sql, params)
            
            # 处理结果
            for result in results:
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                if result.get('keywords'):
                    result['keywords'] = result['keywords'].split(',')
            
            logger.info(f"关键词搜索完成，找到 {len(results)} 个文档")
            return results
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []
    
    def search_documents_by_vector(
        self,
        query_vector: List[float],
        vector_field: str = "document_embedding",
        similarity_threshold: float = 0.5,
        top_k: int = 20,
        additional_filters: str = None
    ) -> List[Dict]:
        """
        基于向量搜索文档
        
        Args:
            query_vector: 查询向量
            vector_field: 向量字段名
            similarity_threshold: 相似度阈值
            top_k: 返回文档数量
            additional_filters: Milvus过滤表达式
        
        Returns:
            List[Dict]: 匹配的文档列表，包含向量相似度信息
        """
        
        try:
            # 在Milvus中搜索
            search_results = self.milvus_client.search(
                collection_name=self.documents_collection_name,
                query_vectors=[query_vector],
                vector_field=vector_field,
                top_k=top_k,
                filters=additional_filters
            )
            
            # 处理搜索结果
            doc_ids = []
            similarity_scores = {}
            
            for result in search_results:
                if result.distance >= similarity_threshold:
                    doc_ids.append(result.id)
                    similarity_scores[result.id] = {
                        "distance": result.distance,
                        "similarity": 1.0 - result.distance,  # 转换为相似度分数
                        "vector_metadata": result.entity
                    }
            
            if not doc_ids:
                logger.info("向量搜索未找到满足阈值的文档")
                return []
            
            # 根据文档ID从MySQL获取完整信息
            placeholders = ','.join(['%s'] * len(doc_ids))
            query_sql = f"""
            SELECT * FROM `{self.db_name}`.`documents`
            WHERE id IN ({placeholders})
            ORDER BY FIELD(id, {','.join(map(str, doc_ids))})
            """
            
            mysql_results = self.mysql_client.fetch_all(query_sql, doc_ids)
            
            # 合并MySQL结果和向量相似度信息
            final_results = []
            for mysql_result in mysql_results:
                doc_id = mysql_result['id']
                if doc_id in similarity_scores:
                    # 处理JSON字段
                    if mysql_result.get('metadata'):
                        mysql_result['metadata'] = json.loads(mysql_result['metadata'])
                    if mysql_result.get('keywords'):
                        mysql_result['keywords'] = mysql_result['keywords'].split(',')
                    
                    # 添加向量搜索信息
                    mysql_result['vector_search'] = similarity_scores[doc_id]
                    final_results.append(mysql_result)
            
            logger.info(f"向量搜索完成，找到 {len(final_results)} 个文档")
            return final_results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def _hybrid_document_search(
        self,
        keywords: str,
        query_vector: List[float],
        vector_field: str,
        mysql_search_fields: List[str],
        similarity_threshold: float,
        top_k: int,
        keyword_weight: float,
        vector_weight: float
    ) -> List[Dict]:
        """混合文档搜索"""
        
        # 分别进行关键词和向量搜索
        keyword_results = self.search_documents_by_keywords(
            keywords, mysql_search_fields, top_k=top_k*2
        )
        
        vector_results = self.search_documents_by_vector(
            query_vector, vector_field, similarity_threshold, top_k=top_k*2
        )
        
        # 合并结果
        all_results = {}
        
        # 处理关键词搜索结果
        for i, result in enumerate(keyword_results):
            doc_id = result['id']
            keyword_score = (len(keyword_results) - i) / len(keyword_results)
            
            all_results[doc_id] = {
                **result,
                "keyword_score": keyword_score,
                "vector_score": 0.0,
                "combined_score": keyword_score * keyword_weight
            }
        
        # 处理向量搜索结果
        for result in vector_results:
            doc_id = result['id']
            vector_score = result['vector_search']['similarity']
            
            if doc_id in all_results:
                # 更新现有记录
                all_results[doc_id]["vector_score"] = vector_score
                all_results[doc_id]["combined_score"] = (
                    all_results[doc_id]["keyword_score"] * keyword_weight +
                    vector_score * vector_weight
                )
                all_results[doc_id]["vector_search"] = result['vector_search']
            else:
                # 新记录
                all_results[doc_id] = {
                    **result,
                    "keyword_score": 0.0,
                    "vector_score": vector_score,
                    "combined_score": vector_score * vector_weight
                }
        
        # 按综合分数排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]
        
        logger.info(f"混合搜索完成，找到 {len(sorted_results)} 个文档")
        return sorted_results
    
    # ======================================
    # Chunk级搜索（第二级搜索）
    # ======================================
    
    def search_chunks_in_document(
        self,
        doc_id: int,
        keywords: str = None,
        query_vector: List[float] = None,
        vector_field: str = "chunk_embedding",
        mysql_search_fields: List[str] = None,
        similarity_threshold: float = 0.5,
        top_k: int = 10,
        combine_results: bool = True,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[Dict]:
        """
        在单个文档中搜索chunks
        
        Args:
            doc_id: 文档ID
            keywords: 搜索关键词
            query_vector: 查询向量
            vector_field: 向量字段名
            mysql_search_fields: MySQL搜索字段
            similarity_threshold: 相似度阈值
            top_k: 返回chunk数量
            combine_results: 是否结合关键词和向量搜索
            keyword_weight: 关键词权重
            vector_weight: 向量权重
        
        Returns:
            List[Dict]: chunk搜索结果列表
        """
        
        # 获取文档信息
        doc_info = self.rag_manager.get_document_info(doc_id)
        if not doc_info:
            logger.warning(f"文档 {doc_id} 不存在")
            return []
        
        chunk_mysql_table = doc_info['chunk_mysql_table']
        chunk_milvus_collection = doc_info['chunk_milvus_collection']
        
        if not chunk_mysql_table or not chunk_milvus_collection:
            logger.warning(f"文档 {doc_id} 的chunk级数据库不存在")
            return []
        
        # 设置默认搜索字段
        if mysql_search_fields is None:
            mysql_search_fields = ["chunk_text", "chunk_title", "keywords"]
        
        # 执行搜索
        try:
            if not combine_results or (not keywords or not query_vector):
                if keywords:
                    return self._search_chunks_by_keywords(
                        chunk_mysql_table, keywords, mysql_search_fields, top_k, doc_id
                    )
                else:
                    return self._search_chunks_by_vector(
                        chunk_milvus_collection, chunk_mysql_table, query_vector, 
                        vector_field, similarity_threshold, top_k, doc_id
                    )
            else:
                return self._hybrid_chunk_search(
                    chunk_mysql_table, chunk_milvus_collection, keywords, query_vector,
                    vector_field, mysql_search_fields, similarity_threshold, top_k,
                    keyword_weight, vector_weight, doc_id
                )
                
        except Exception as e:
            logger.error(f"在文档 {doc_id} 中搜索chunks失败: {e}")
            return []
    
    def search_chunks_parallel(
        self,
        doc_ids: List[int],
        keywords: str = None,
        query_vector: List[float] = None,
        vector_field: str = "chunk_embedding",
        mysql_search_fields: List[str] = None,
        similarity_threshold: float = 0.5,
        top_k_per_doc: int = 5,
        max_total_results: int = 50,
        combine_results: bool = True,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> Dict[int, List[Dict]]:
        """
        并行搜索多个文档的chunks
        
        Args:
            doc_ids: 文档ID列表
            keywords: 搜索关键词
            query_vector: 查询向量
            vector_field: 向量字段名
            mysql_search_fields: MySQL搜索字段
            similarity_threshold: 相似度阈值
            top_k_per_doc: 每个文档返回的chunk数量
            max_total_results: 总结果数量限制
            combine_results: 是否结合搜索结果
            keyword_weight: 关键词权重
            vector_weight: 向量权重
        
        Returns:
            Dict[int, List[Dict]]: {doc_id: [chunk_results]}
        """
        
        if not doc_ids:
            return {}
        
        logger.info(f"开始并行搜索 {len(doc_ids)} 个文档的chunks")
        
        # 使用线程池并行执行搜索
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(doc_ids), 10)) as executor:
            # 提交搜索任务
            future_to_doc = {
                executor.submit(
                    self.search_chunks_in_document,
                    doc_id, keywords, query_vector, vector_field,
                    mysql_search_fields, similarity_threshold, top_k_per_doc,
                    combine_results, keyword_weight, vector_weight
                ): doc_id
                for doc_id in doc_ids
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_doc):
                doc_id = future_to_doc[future]
                try:
                    chunk_results = future.result()
                    if chunk_results:
                        results[doc_id] = chunk_results
                        logger.info(f"文档 {doc_id} 搜索完成，找到 {len(chunk_results)} 个chunks")
                    else:
                        logger.info(f"文档 {doc_id} 未找到相关chunks")
                except Exception as e:
                    logger.error(f"文档 {doc_id} 搜索失败: {e}")
        
        # 如果需要限制总结果数量，进行全局排序
        if max_total_results and max_total_results < sum(len(chunks) for chunks in results.values()):
            results = self._limit_total_chunk_results(results, max_total_results)
        
        logger.info(f"并行搜索完成，共 {len(results)} 个文档有结果")
        return results
    
    def _search_chunks_by_keywords(
        self, 
        table_name: str, 
        keywords: str, 
        search_fields: List[str], 
        top_k: int,
        doc_id: int
    ) -> List[Dict]:
        """在chunk表中进行关键词搜索"""
        
        # 构建搜索SQL
        where_conditions = []
        params = []
        
        keyword_conditions = []
        for field in search_fields:
            if field in ["chunk_text", "keywords"]:
                # 使用全文索引
                keyword_conditions.append(f"MATCH(`{field}`) AGAINST(%s IN NATURAL LANGUAGE MODE)")
                params.append(keywords)
            else:
                # 使用LIKE搜索
                keyword_conditions.append(f"`{field}` LIKE %s")
                params.append(f"%{keywords}%")
        
        if keyword_conditions:
            where_conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        query_sql = f"""
        SELECT * FROM `{self.db_name}`.`{table_name}`
        WHERE {where_clause}
        ORDER BY chunk_index ASC
        LIMIT %s
        """
        params.append(top_k)
        
        results = self.mysql_client.fetch_all(query_sql, params)
        
        # 添加文档ID信息
        for result in results:
            result['document_id'] = doc_id
            if result.get('metadata'):
                result['metadata'] = json.loads(result['metadata'])
        
        return results
    
    def _search_chunks_by_vector(
        self,
        collection_name: str,
        table_name: str,
        query_vector: List[float],
        vector_field: str,
        similarity_threshold: float,
        top_k: int,
        doc_id: int
    ) -> List[Dict]:
        """在chunk向量集合中进行向量搜索"""
        
        # 在Milvus中搜索
        search_results = self.milvus_client.search(
            collection_name=collection_name,
            query_vectors=[query_vector],
            vector_field=vector_field,
            top_k=top_k
        )
        
        # 过滤相似度结果
        chunk_ids = []
        similarity_scores = {}
        
        for result in search_results:
            if result.distance >= similarity_threshold:
                chunk_ids.append(result.id)
                similarity_scores[result.id] = {
                    "distance": result.distance,
                    "similarity": 1.0 - result.distance,
                    "vector_metadata": result.entity
                }
        
        if not chunk_ids:
            return []
        
        # 从MySQL获取完整chunk信息
        placeholders = ','.join(['%s'] * len(chunk_ids))
        query_sql = f"""
        SELECT * FROM `{self.db_name}`.`{table_name}`
        WHERE id IN ({placeholders})
        ORDER BY chunk_index ASC
        """
        
        mysql_results = self.mysql_client.fetch_all(query_sql, chunk_ids)
        
        # 合并结果
        final_results = []
        for mysql_result in mysql_results:
            chunk_id = mysql_result['id']
            if chunk_id in similarity_scores:
                if mysql_result.get('metadata'):
                    mysql_result['metadata'] = json.loads(mysql_result['metadata'])
                
                mysql_result['document_id'] = doc_id
                mysql_result['vector_search'] = similarity_scores[chunk_id]
                final_results.append(mysql_result)
        
        return final_results
    
    def _hybrid_chunk_search(
        self,
        table_name: str,
        collection_name: str,
        keywords: str,
        query_vector: List[float],
        vector_field: str,
        mysql_search_fields: List[str],
        similarity_threshold: float,
        top_k: int,
        keyword_weight: float,
        vector_weight: float,
        doc_id: int
    ) -> List[Dict]:
        """混合chunk搜索"""
        
        # 分别进行关键词和向量搜索
        keyword_results = self._search_chunks_by_keywords(
            table_name, keywords, mysql_search_fields, top_k*2, doc_id
        )
        
        vector_results = self._search_chunks_by_vector(
            collection_name, table_name, query_vector, vector_field,
            similarity_threshold, top_k*2, doc_id
        )
        
        # 合并结果
        all_results = {}
        
        # 处理关键词结果
        for i, result in enumerate(keyword_results):
            chunk_id = result['id']
            keyword_score = (len(keyword_results) - i) / len(keyword_results)
            
            all_results[chunk_id] = {
                **result,
                "keyword_score": keyword_score,
                "vector_score": 0.0,
                "combined_score": keyword_score * keyword_weight
            }
        
        # 处理向量结果
        for result in vector_results:
            chunk_id = result['id']
            vector_score = result['vector_search']['similarity']
            
            if chunk_id in all_results:
                all_results[chunk_id]["vector_score"] = vector_score
                all_results[chunk_id]["combined_score"] = (
                    all_results[chunk_id]["keyword_score"] * keyword_weight +
                    vector_score * vector_weight
                )
                all_results[chunk_id]["vector_search"] = result['vector_search']
            else:
                all_results[chunk_id] = {
                    **result,
                    "keyword_score": 0.0,
                    "vector_score": vector_score,
                    "combined_score": vector_score * vector_weight
                }
        
        # 排序并返回
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]
        
        return sorted_results
    
    def _limit_total_chunk_results(self, results: Dict[int, List[Dict]], max_total: int) -> Dict[int, List[Dict]]:
        """限制总的chunk结果数量"""
        
        # 收集所有结果并按分数排序
        all_chunks = []
        for doc_id, chunks in results.items():
            for chunk in chunks:
                chunk['source_doc_id'] = doc_id
                all_chunks.append(chunk)
        
        # 按综合分数或相似度分数排序
        all_chunks.sort(key=lambda x: x.get('combined_score', x.get('vector_search', {}).get('similarity', 0)), reverse=True)
        
        # 取前max_total个结果
        top_chunks = all_chunks[:max_total]
        
        # 重新组织结果
        limited_results = {}
        for chunk in top_chunks:
            doc_id = chunk['source_doc_id']
            del chunk['source_doc_id']  # 移除临时字段
            
            if doc_id not in limited_results:
                limited_results[doc_id] = []
            limited_results[doc_id].append(chunk)
        
        return limited_results
    
    # ======================================
    # 便捷的完整搜索流程
    # ======================================
    
    def full_search(
        self,
        keywords: str = None,
        query_vector: List[float] = None,
        document_vector_field: str = "document_embedding",
        chunk_vector_field: str = "chunk_embedding",
        doc_top_k: int = 10,
        chunk_top_k_per_doc: int = 5,
        max_total_chunks: int = 50,
        doc_similarity_threshold: float = 0.5,
        chunk_similarity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        完整的两级搜索流程
        
        Args:
            keywords: 搜索关键词
            query_vector: 查询向量
            document_vector_field: 文档级向量字段
            chunk_vector_field: chunk级向量字段
            doc_top_k: 文档级搜索返回数量
            chunk_top_k_per_doc: 每个文档的chunk搜索数量
            max_total_chunks: 总chunk结果限制
            doc_similarity_threshold: 文档级相似度阈值
            chunk_similarity_threshold: chunk级相似度阈值
        
        Returns:
            Dict: 完整搜索结果
            {
                "relevant_documents": [...],      # 相关文档列表
                "chunk_results": {doc_id: [...]}, # chunk搜索结果
                "search_stats": {...}             # 搜索统计信息
            }
        """
        
        logger.info("开始完整的两级RAG搜索")
        
        # 第一步：文档级搜索
        logger.info("步骤1：文档级搜索")
        relevant_docs = self.search_documents(
            keywords=keywords,
            query_vector=query_vector,
            vector_field=document_vector_field,
            similarity_threshold=doc_similarity_threshold,
            top_k=doc_top_k
        )
        
        if not relevant_docs:
            logger.info("未找到相关文档")
            return {
                "relevant_documents": [],
                "chunk_results": {},
                "search_stats": {
                    "doc_count": 0,
                    "chunk_count": 0,
                    "total_time": 0
                }
            }
        
        doc_ids = [doc['id'] for doc in relevant_docs]
        logger.info(f"找到 {len(relevant_docs)} 个相关文档: {doc_ids}")
        
        # 第二步：chunk级并行搜索
        logger.info("步骤2：chunk级并行搜索")
        chunk_results = self.search_chunks_parallel(
            doc_ids=doc_ids,
            keywords=keywords,
            query_vector=query_vector,
            vector_field=chunk_vector_field,
            similarity_threshold=chunk_similarity_threshold,
            top_k_per_doc=chunk_top_k_per_doc,
            max_total_results=max_total_chunks
        )
        
        # 统计信息
        total_chunks = sum(len(chunks) for chunks in chunk_results.values())
        
        search_stats = {
            "doc_count": len(relevant_docs),
            "chunk_count": total_chunks,
            "docs_with_chunks": len(chunk_results)
        }
        
        logger.info(f"搜索完成: {search_stats}")
        
        return {
            "relevant_documents": relevant_docs,
            "chunk_results": chunk_results,
            "search_stats": search_stats
        }