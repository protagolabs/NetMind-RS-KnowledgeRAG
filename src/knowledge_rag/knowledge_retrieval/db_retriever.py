"""
数据库检索器 - 实现知识库的多种搜索功能
===========================================

【整体架构说明】
这是一个基于RAG（Retrieval-Augmented Generation）的知识检索系统，采用"混合搜索"策略：
- 向量搜索：基于语义相似度，能理解内容含义，适合概念性查询
- 全文搜索：基于关键词匹配，精确但缺乏语义理解，适合具体词汇查询
- 两阶段检索：先粗粒度（文档级）再细粒度（chunk级），平衡效率与精度

【数据库设计】
系统使用双数据库架构：
1. MySQL - 存储结构化数据和支持全文检索
   - rag_documents表：存储文档基本信息、摘要、关键词等
   - chunks_[source_id]表：存储每个文档的chunk（文档片段）数据
   
2. Milvus - 专门的向量数据库，支持高效向量相似度搜索
   - documents_vectors集合：存储文档级向量（summary_embedding, key_words_embedding）
   - documents_insights_vectors集合：存储文档的insights向量
   - chunks_vectors_[source_id]集合：存储每个文档的chunk向量

【核心检索功能】
1. 文档级向量搜索 - 在文档的summary和key_words向量中搜索
2. 文档级insights向量搜索 - 在文档的insights向量中搜索  
3. 文档级全文搜索 - 在文档的文本内容中搜索
4. Chunk级向量搜索 - 在特定文档的chunks中搜索
5. Chunk级全文搜索 - 在chunks的文本内容中搜索

【智能检索流程】
1. 先在文档级别搜索相关文档（粗筛选）
2. 根据相关文档确定要搜索的chunk集合（缩小范围）  
3. 并行在这些chunk集合中进行详细搜索（精确匹配）
4. 返回按相似度/相关度排序的结果

【技术原理】
- 向量搜索：使用L2距离计算向量相似度，距离越小相似度越高
- 全文搜索：使用MySQL的MATCH...AGAINST自然语言模式
- 相似度转换：L2距离转换为0-1相似度分数：1/(1+distance)
- 并行处理：多个数据源同时搜索，提高检索效率

作者: XYZ-Algorithm-Team
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import mysql.connector
from pymilvus import connections, Collection
import numpy as np

logger = logging.getLogger(__name__)

class DBRetriever:
    """
    数据库检索器 - 支持多种搜索模式
    ==============================
    
    【核心职责】
    这个类是整个知识检索系统的核心引擎，负责：
    1. 管理MySQL和Milvus双数据库连接
    2. 实现多种检索策略（向量搜索、全文搜索、混合搜索）
    3. 处理检索结果的合并、排序和过滤
    4. 提供统一的检索接口
    
    【设计模式】
    采用"策略模式"，根据查询类型选择最适合的搜索策略：
    - 概念性查询 → 向量搜索（理解语义）
    - 具体词汇查询 → 全文搜索（精确匹配）
    - 复杂查询 → 混合搜索（多策略结合）
    """
    
    def __init__(self, 
                 mysql_config: Dict[str, Any],
                 milvus_config: Dict[str, Any]):
        """
        初始化检索器 - 建立双数据库连接
        
        【初始化流程】
        1. 保存数据库配置信息
        2. 初始化连接状态变量
        3. 执行数据库连接操作
        4. 验证连接是否成功
        
        Args:
            mysql_config: MySQL数据库配置字典
                - host: 数据库主机地址
                - port: 端口号（默认3306）
                - user: 用户名
                - password: 密码
                - database: 数据库名
                - charset: 字符集（推荐utf8mb4支持emoji）
                
            milvus_config: Milvus向量数据库配置字典
                - host: Milvus服务地址
                - port: 端口号（默认19530）
                - alias: 连接别名（用于管理多个连接）
        
        【为什么用两个数据库？】
        - MySQL：擅长结构化数据存储和全文检索，但向量搜索性能差
        - Milvus：专门针对向量搜索优化，但不擅长结构化数据管理
        - 两者结合：发挥各自优势，实现高效的混合检索
        """
        self.mysql_config = mysql_config
        self.milvus_config = milvus_config
        
        # 数据库连接状态管理
        self.mysql_conn = None          # MySQL连接对象
        self.milvus_connected = False   # Milvus连接状态标志
        
        # 立即建立数据库连接（失败会抛出异常）
        self._connect_databases()
    
    def _connect_databases(self):
        """
        连接到MySQL和Milvus数据库
        
        【连接策略】
        1. 使用连接池管理MySQL连接，提高并发性能
        2. 启用autocommit避免事务管理复杂性（适合读多写少场景）
        3. 使用UTF8MB4字符集支持emoji和特殊字符
        4. Milvus使用别名管理，支持多实例连接
        
        【错误处理】
        任何连接失败都会抛出异常，确保系统不会在不完整状态下运行
        """
        try:
            # 连接MySQL - 关系型数据库，存储结构化数据
            self.mysql_conn = mysql.connector.connect(
                host=self.mysql_config['host'],           # 数据库服务器地址
                port=self.mysql_config['port'],           # MySQL端口
                user=self.mysql_config['user'],           # 用户名
                password=self.mysql_config['password'],   # 密码
                database=self.mysql_config['database'],   # 目标数据库
                charset=self.mysql_config['charset'],     # 字符集（支持emoji）
                autocommit=True  # 自动提交，简化事务管理（适合读多写少场景）
            )
            logger.info("已连接到MySQL数据库")
            
            # 连接Milvus - 向量数据库，专门用于向量相似度搜索
            connections.connect(
                alias=self.milvus_config['alias'],  # 连接别名，便于管理多个连接
                host=self.milvus_config['host'],    # Milvus服务地址
                port=self.milvus_config['port']     # Milvus端口
            )
            self.milvus_connected = True
            logger.info("已连接到Milvus向量数据库")
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            # 连接失败立即抛出异常，避免系统在不完整状态下运行
            raise
    
    def search_documents_by_vector(
        self, 
        query_vector: List[float],
        vector_field: str = "summary_embedding",
        top_k: int = 10,
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        文档级向量搜索 - 基于语义相似度查找相关文档
        ===========================================
        
        【工作原理】
        1. 将查询向量与文档向量集合进行相似度计算
        2. 使用L2距离度量（欧几里得距离），距离越小相似度越高
        3. 从Milvus获取最相似的向量及其对应的file_id
        4. 根据file_id从MySQL获取完整的文档信息
        5. 合并结果并按相似度排序
        
        【向量搜索优势】
        - 理解语义：能找到意思相近但用词不同的内容
        - 多语言支持：向量表示跨越语言障碍
        - 模糊匹配：即使查询不精确也能找到相关内容
        - 主题发现：能发现隐含的主题关联
        
        【参数说明】
        Args:
            query_vector: 查询向量（通常由embedding模型生成）
                - 维度必须与数据库中存储的向量维度一致
                - 通常是512、768、1024等维度的浮点数组
                
            vector_field: 要搜索的向量字段名
                - "summary_embedding": 文档摘要的向量表示（推荐，语义丰富）
                - "key_words_embedding": 关键词的向量表示（精确性高）
                
            top_k: 返回前K个最相似的结果
                - 建议5-20，过多会影响相关性
                - 会在后续filtering中进一步筛选
                
            threshold: 相似度阈值（0-1之间）
                - 0.8较严格，只返回高度相关的内容
                - 0.6较宽松，可能包含部分相关的内容
                - 根据具体应用场景调整
                
        Returns:
            List[Dict]: 搜索结果列表，每个元素包含：
                - 文档基本信息（id, file_id, file_name等）
                - similarity_score: 相似度分数（0-1）
                - search_field: 搜索的字段名
                - 解析后的JSON字段（insights、key_words）
        
        【技术细节】
        - 使用L2距离：sqrt(sum((a_i - b_i)^2))，适合大多数embedding
        - 相似度转换：1/(1+distance)，将距离转换为0-1分数
        - nprobe=10：搜索参数，平衡精度与速度
        """
        try:
            # 第一步：在Milvus向量数据库中进行相似度搜索
            collection = Collection("documents_vectors")  # 文档向量集合
            
            # 配置搜索参数
            search_params = {
                "metric_type": "L2",     # 使用L2距离（欧几里得距离）
                "params": {"nprobe": 10} # 搜索聚类数量，影响精度vs速度平衡
            }
            
            # 执行向量搜索
            results = collection.search(
                data=[query_vector],        # 查询向量（列表格式，支持批量查询）
                anns_field=vector_field,    # 要搜索的向量字段
                param=search_params,        # 搜索参数
                limit=top_k,               # 返回最相似的top_k个结果
                output_fields=["file_id"]   # 需要返回的额外字段
            )
            
            # 第二步：处理向量搜索结果，获取文档详细信息
            doc_results = []
            if results and len(results[0]) > 0:  # 检查是否有搜索结果
                # 提取所有匹配文档的file_id
                file_ids = [hit.entity.get("file_id") for hit in results[0]]
                
                # 第三步：从MySQL获取文档的完整结构化信息
                # 使用参数化查询防止SQL注入
                placeholders = ', '.join(['%s'] * len(file_ids))
                query_sql = f"""
                SELECT id, file_id, file_name, summary, insights, key_words, 
                       processing_status, created_at
                FROM rag_documents 
                WHERE file_id IN ({placeholders})
                """
                
                cursor = self.mysql_conn.cursor(dictionary=True)  # 返回字典格式结果
                cursor.execute(query_sql, file_ids)
                docs = cursor.fetchall()
                cursor.close()
                
                # 第四步：合并向量搜索结果和MySQL数据库信息
                # 创建file_id到文档信息的映射，便于快速查找
                docs_dict = {doc['file_id']: doc for doc in docs}
                
                # 遍历向量搜索结果，补充完整信息
                for hit in results[0]:
                    file_id = hit.entity.get("file_id")
                    if file_id in docs_dict:  # 确保MySQL中存在对应记录
                        doc_info = docs_dict[file_id]
                        
                        # 【关键】将L2距离转换为相似度分数
                        # 公式：1/(1+distance) 确保分数在0-1之间，距离越小分数越高
                        doc_info['similarity_score'] = float(1.0 / (1.0 + hit.distance))
                        doc_info['search_field'] = vector_field  # 记录搜索字段
                        
                        # 第五步：解析存储为JSON字符串的字段
                        # insights和key_words在MySQL中以JSON格式存储
                        if doc_info.get('insights') and isinstance(doc_info['insights'], str):
                            doc_info['insights'] = json.loads(doc_info['insights'])
                        if doc_info.get('key_words') and isinstance(doc_info['key_words'], str):
                            doc_info['key_words'] = json.loads(doc_info['key_words'])
                        
                        # 第六步：应用相似度阈值过滤
                        # 只返回相似度高于阈值的结果，提高结果质量
                        if doc_info['similarity_score'] >= threshold:
                            doc_results.append(doc_info)
            
            logger.info(f"文档向量搜索完成，返回{len(doc_results)}个结果")
            return doc_results
            
        except Exception as e:
            logger.error(f"文档向量搜索失败: {e}")
            return []
    
    def search_documents_by_insights(self, 
                                   query_vector: List[float],
                                   top_k: int = 10,
                                   threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        文档级insights向量搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回前K个结果
            threshold: 相似度阈值
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 在insights向量集合中搜索
            collection = Collection("documents_insights_vectors")
            
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = collection.search(
                data=[query_vector],
                anns_field="insight_embedding",
                param=search_params,
                limit=top_k,
                output_fields=["file_id", "insight_text", "insight_index"]
            )
            
            # 获取文档详细信息
            doc_results = []
            if results and len(results[0]) > 0:
                file_ids = list(set([hit.entity.get("file_id") for hit in results[0]]))
                
                # 从MySQL获取文档详细信息
                placeholders = ', '.join(['%s'] * len(file_ids))
                query_sql = f"""
                SELECT id, file_id, file_name, summary, insights, key_words, 
                       processing_status, created_at
                FROM rag_documents 
                WHERE file_id IN ({placeholders})
                """
                
                cursor = self.mysql_conn.cursor(dictionary=True)
                cursor.execute(query_sql, file_ids)
                docs = cursor.fetchall()
                cursor.close()
                
                # 合并结果
                docs_dict = {doc['file_id']: doc for doc in docs}
                
                for hit in results[0]:
                    file_id = hit.entity.get("file_id")
                    if file_id in docs_dict:
                        doc_info = docs_dict[file_id].copy()
                        doc_info['similarity_score'] = float(1.0 / (1.0 + hit.distance))
                        doc_info['matched_insight'] = hit.entity.get("insight_text")
                        doc_info['insight_index'] = hit.entity.get("insight_index")
                        doc_info['search_field'] = "insights"
                        
                        # 解析JSON字段
                        if doc_info.get('insights') and isinstance(doc_info['insights'], str):
                            doc_info['insights'] = json.loads(doc_info['insights'])
                        if doc_info.get('key_words') and isinstance(doc_info['key_words'], str):
                            doc_info['key_words'] = json.loads(doc_info['key_words'])
                        
                        if doc_info['similarity_score'] >= threshold:
                            doc_results.append(doc_info)
            
            logger.info(f"文档insights搜索完成，返回{len(doc_results)}个结果")
            return doc_results
            
        except Exception as e:
            logger.error(f"文档insights搜索失败: {e}")
            return []
    
    def search_documents_by_text(self, 
                               query_text: str,
                               search_fields: List[str] = ["summary", "doc_markdown_content"],
                               top_k: int = 10) -> List[Dict[str, Any]]:
        """
        文档级全文搜索 - 基于关键词匹配查找相关文档
        ============================================
        
        【工作原理】
        使用MySQL的全文索引功能进行文本搜索：
        1. 对查询文本进行词汇分析和处理
        2. 在指定字段中查找包含查询词汇的文档
        3. 计算文本相关度分数（基于词频、位置等因素）
        4. 按相关度排序返回结果
        
        【全文搜索 vs 向量搜索】
        全文搜索优势：
        - 精确匹配：能准确找到包含特定词汇的内容
        - 速度快：基于索引，查询效率高
        - 词汇敏感：适合专业术语、人名、地名等精确查询
        - 无需预处理：直接对原始文本搜索
        
        全文搜索局限：
        - 缺乏语义理解：无法理解同义词、近义词
        - 严格匹配：查询词必须在文档中出现
        - 语言依赖：对不同语言支持程度不同
        
        【参数说明】
        Args:
            query_text: 查询文本字符串
                - 支持自然语言查询
                - MySQL会自动处理停用词（如"的"、"是"等）
                - 推荐使用关键词组合，而非完整句子
                
            search_fields: 要搜索的字段列表
                - "summary": 文档摘要（推荐，内容精炼）
                - "doc_markdown_content": 文档完整内容（覆盖全面）
                - 可同时搜索多个字段，提高召回率
                
            top_k: 返回前K个最相关的结果
                - 基于MySQL的相关度评分排序
                - 建议10-50，根据应用需求调整
                
        Returns:
            List[Dict]: 搜索结果列表，每个元素包含：
                - 文档完整信息
                - relevance_score: MySQL计算的相关度分数
                - search_field: 标识为"fulltext"搜索
        
        【技术细节】
        - 使用MATCH...AGAINST NATURAL LANGUAGE MODE
        - 相关度计算考虑词频、文档长度、词汇稀有度等因素
        - 需要在搜索字段上建立FULLTEXT索引
        """
        try:
            # 第一步：动态构建全文搜索条件
            # 为每个搜索字段生成MATCH...AGAINST条件
            search_conditions = []
            for field in search_fields:
                # NATURAL LANGUAGE MODE：自然语言搜索模式
                # MySQL会自动分析查询文本，提取关键词，计算相关度
                search_conditions.append(f"MATCH({field}) AGAINST(%s IN NATURAL LANGUAGE MODE)")
            
            # 第二步：构建完整的SQL查询语句
            search_sql = f"""
            SELECT id, file_id, file_name, summary, insights, key_words, 
                   processing_status, created_at,
                   ({' + '.join(search_conditions)}) as relevance_score
            FROM rag_documents 
            WHERE {' OR '.join(search_conditions)}
            ORDER BY relevance_score DESC
            LIMIT %s
            """
            # 注意：
            # - SELECT中计算总相关度：各字段相关度之和
            # - WHERE中使用OR：任一字段匹配即返回该文档
            # - ORDER BY：按相关度降序排列，最相关的在前
            
            # 第三步：准备查询参数
            # 需要为每个MATCH条件提供查询文本，所以要重复query_text
            cursor = self.mysql_conn.cursor(dictionary=True)
            params = [query_text] * (len(search_fields) * 2) + [top_k]
            # 参数解释：
            # - query_text重复2倍字段数：SELECT和WHERE中各用一次
            # - 最后加上LIMIT的top_k值
            
            # 第四步：执行搜索
            cursor.execute(search_sql, params)
            results = cursor.fetchall()
            cursor.close()
            
            # 第五步：后处理搜索结果
            for result in results:
                # 解析JSON格式存储的字段
                if result.get('insights') and isinstance(result['insights'], str):
                    result['insights'] = json.loads(result['insights'])
                if result.get('key_words') and isinstance(result['key_words'], str):
                    result['key_words'] = json.loads(result['key_words'])
                
                # 标记搜索方式，便于后续处理区分
                result['search_field'] = "fulltext"
            
            logger.info(f"文档全文搜索完成，返回{len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"文档全文搜索失败: {e}")
            return []
    
    def get_document_by_file_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        根据file_id获取文档详细信息
        
        Args:
            file_id: 文档ID
            
        Returns:
            Dict: 文档信息，如果不存在返回None
        """
        try:
            cursor = self.mysql_conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, file_id, file_name, summary, insights, key_words, 
                       doc_markdown_content, chunk_mysql_table, chunk_milvus_collection,
                       chunk_count, processing_status, created_at, updated_at
                FROM rag_documents 
                WHERE file_id = %s
            """, (file_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                # 解析JSON字段
                if result.get('insights') and isinstance(result['insights'], str):
                    result['insights'] = json.loads(result['insights'])
                if result.get('key_words') and isinstance(result['key_words'], str):
                    result['key_words'] = json.loads(result['key_words'])
            
            return result
            
        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            return None
    
    def search_chunks_by_vector(self,
                               source_ids: List[str],
                               query_vector: List[float],
                               vector_field: str = "summary_embedding",
                               top_k: int = 20,
                               threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        在指定文档的chunks中进行向量搜索
        
        Args:
            source_ids: 要搜索的文档source_id列表
            query_vector: 查询向量
            vector_field: 向量字段名
            top_k: 每个集合返回的结果数
            threshold: 相似度阈值
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            all_results = []
            
            for source_id in source_ids:
                try:
                    # 生成集合名
                    clean_source_id = source_id.replace('-', '_').replace('.', '_')
                    collection_name = f"chunks_vectors_{clean_source_id}"
                    
                    # 检查集合是否存在
                    from pymilvus import utility
                    if not utility.has_collection(collection_name):
                        logger.warning(f"集合 {collection_name} 不存在，跳过")
                        continue
                    
                    collection = Collection(collection_name)
                    
                    search_params = {
                        "metric_type": "L2",
                        "params": {"nprobe": 10}
                    }
                    
                    results = collection.search(
                        data=[query_vector],
                        anns_field=vector_field,
                        param=search_params,
                        limit=top_k,
                        output_fields=["chunk_id", "source_id"]
                    )
                    
                    if results and len(results[0]) > 0:
                        # 获取chunk详细信息
                        chunk_ids = [hit.entity.get("chunk_id") for hit in results[0]]
                        table_name = f"chunks_{clean_source_id}"
                        
                        placeholders = ', '.join(['%s'] * len(chunk_ids))
                        query_sql = f"""
                        SELECT id, source_id, chunk_id, summary, insights, key_words, 
                               chunk_markdown_content, created_at
                        FROM {table_name}
                        WHERE chunk_id IN ({placeholders})
                        """
                        
                        cursor = self.mysql_conn.cursor(dictionary=True)
                        cursor.execute(query_sql, chunk_ids)
                        chunks = cursor.fetchall()
                        cursor.close()
                        
                        # 合并结果
                        chunks_dict = {chunk['chunk_id']: chunk for chunk in chunks}
                        
                        for hit in results[0]:
                            chunk_id = hit.entity.get("chunk_id")
                            if chunk_id in chunks_dict:
                                chunk_info = chunks_dict[chunk_id]
                                chunk_info['similarity_score'] = float(1.0 / (1.0 + hit.distance))
                                chunk_info['search_field'] = vector_field
                                
                                # 解析JSON字段
                                if chunk_info.get('insights') and isinstance(chunk_info['insights'], str):
                                    chunk_info['insights'] = json.loads(chunk_info['insights'])
                                if chunk_info.get('key_words') and isinstance(chunk_info['key_words'], str):
                                    chunk_info['key_words'] = json.loads(chunk_info['key_words'])
                                
                                if chunk_info['similarity_score'] >= threshold:
                                    all_results.append(chunk_info)
                
                except Exception as e:
                    logger.error(f"搜索source_id {source_id}的chunks失败: {e}")
                    continue
            
            # 按相似度排序
            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"chunk向量搜索完成，返回{len(all_results)}个结果")
            return all_results
            
        except Exception as e:
            logger.error(f"chunk向量搜索失败: {e}")
            return []
    
    def search_chunks_by_text(self,
                             source_ids: List[str],
                             query_text: str,
                             search_fields: List[str] = ["summary", "chunk_markdown_content"],
                             top_k: int = 20) -> List[Dict[str, Any]]:
        """
        在指定文档的chunks中进行全文搜索
        
        Args:
            source_ids: 要搜索的文档source_id列表
            query_text: 查询文本
            search_fields: 搜索字段列表
            top_k: 每个表返回的结果数
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            all_results = []
            
            for source_id in source_ids:
                try:
                    # 生成表名
                    clean_source_id = source_id.replace('-', '_').replace('.', '_')
                    table_name = f"chunks_{clean_source_id}"
                    
                    # 检查表是否存在
                    cursor = self.mysql_conn.cursor(dictionary=True)
                    cursor.execute("""
                        SELECT COUNT(*) as count FROM information_schema.tables 
                        WHERE table_schema = %s AND table_name = %s
                    """, (self.mysql_config['database'], table_name))
                    
                    if cursor.fetchone()['count'] == 0:
                        logger.warning(f"表 {table_name} 不存在，跳过")
                        cursor.close()
                        continue
                    
                    # 构建全文搜索SQL
                    search_conditions = []
                    for field in search_fields:
                        search_conditions.append(f"MATCH({field}) AGAINST(%s IN NATURAL LANGUAGE MODE)")
                    
                    search_sql = f"""
                    SELECT id, source_id, chunk_id, summary, insights, key_words, 
                           chunk_markdown_content, created_at,
                           ({' + '.join(search_conditions)}) as relevance_score
                    FROM {table_name}
                    WHERE {' OR '.join(search_conditions)}
                    ORDER BY relevance_score DESC
                    LIMIT %s
                    """
                    
                    # 执行搜索
                    params = [query_text] * (len(search_fields) * 2) + [top_k]
                    cursor.execute(search_sql, params)
                    results = cursor.fetchall()
                    cursor.close()
                    
                    # 处理结果
                    for result in results:
                        # 解析JSON字段
                        if result.get('insights') and isinstance(result['insights'], str):
                            result['insights'] = json.loads(result['insights'])
                        if result.get('key_words') and isinstance(result['key_words'], str):
                            result['key_words'] = json.loads(result['key_words'])
                        
                        result['search_field'] = "fulltext"
                        result['search_table'] = table_name
                        all_results.append(result)
                
                except Exception as e:
                    logger.error(f"搜索表 {table_name} 失败: {e}")
                    continue
            
            # 按相关度排序
            all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            logger.info(f"chunk全文搜索完成，返回{len(all_results)}个结果")
            return all_results
            
        except Exception as e:
            logger.error(f"chunk全文搜索失败: {e}")
            return []
    
    def comprehensive_search(self,
                           query_vector: List[float],
                           query_text: str = None,
                           doc_top_k: int = 5,
                           chunk_top_k: int = 10,
                           final_top_k: int = 20) -> Dict[str, Any]:
        """
        综合搜索 - 智能两阶段检索策略
        ============================
        
        【核心思想】
        这是整个系统最核心的检索策略，采用"从粗到细"的两阶段方法：
        1. 文档级粗筛选：快速确定相关文档集合
        2. chunk级精搜索：在相关文档内部精确查找答案
        
        【为什么用两阶段搜索？】
        单一搜索策略的局限性：
        - 只搜文档：粒度太粗，难以定位具体信息
        - 只搜chunk：搜索范围太大，效率低且噪音多
        - 直接全局chunk搜索：计算量大，相关性可能较差
        
        两阶段搜索优势：
        - 效率高：先缩小搜索范围，再精确查找
        - 精度高：结合文档相关性和chunk具体性
        - 可控性：可以调整各阶段的参数
        - 可解释：能追踪搜索路径（哪个文档->哪个chunk）
        
        【搜索策略组合】
        同时使用多种搜索方法，提高召回率：
        - 向量搜索：捕获语义相关内容
        - insights搜索：利用文档核心观点
        - 全文搜索：精确匹配关键词（可选）
        
        【参数说明】
        Args:
            query_vector: 查询向量（必需）
                - 由embedding模型生成
                - 用于语义相似度计算
                
            query_text: 查询文本（可选）
                - 如果提供，会启用全文搜索
                - 建议提供，能提高搜索覆盖度
                
            doc_top_k: 文档级搜索返回数量
                - 控制第一阶段搜索范围
                - 建议3-10，太多会引入噪音，太少会遗漏相关内容
                
            chunk_top_k: 每个文档的chunk搜索数量
                - 在每个相关文档中搜索的chunk数量
                - 建议5-20，根据文档大小调整
                
            final_top_k: 最终返回的chunk数量
                - 合并所有结果后的最终返回数量
                - 建议10-50，根据应用需求调整
                
        Returns:
            Dict: 完整的搜索结果包含：
                - relevant_documents: 相关文档列表（文档级结果）
                - relevant_chunks: 相关chunk列表（最终答案候选）
                - search_summary: 搜索过程统计信息
        
        【技术优化】
        - 并行搜索：多种搜索策略同时进行
        - 智能去重：基于ID去重，保留最高分数
        - 多维排序：综合相似度和相关度分数
        - 阈值过滤：只保留高质量结果
        """
        # 初始化搜索结果结构
        result = {
            'relevant_documents': [],    # 第一阶段：相关文档列表
            'relevant_chunks': [],       # 第二阶段：相关chunk列表（最终答案候选）
            'search_summary': {}         # 搜索过程统计和元信息
        }
        
        try:
            # ============ 第一阶段：文档级多策略搜索 ============
            logger.info("开始文档级搜索...")
            
            # 策略1：向量搜索文档摘要
            # 基于文档摘要向量进行语义相似度搜索
            doc_vector_results = self.search_documents_by_vector(
                query_vector, "summary_embedding", doc_top_k
            )
            
            # 策略2：insights向量搜索
            # 在文档的核心观点/洞察中搜索相关内容
            doc_insights_results = self.search_documents_by_insights(
                query_vector, doc_top_k
            )
            
            # 策略3：全文搜索（条件性启用）
            # 如果提供了查询文本，启用关键词精确匹配
            doc_text_results = []
            if query_text:
                doc_text_results = self.search_documents_by_text(
                    query_text, top_k=doc_top_k
                )
            
            # ============ 文档结果合并与去重 ============
            # 使用字典结构按file_id去重，保留最高分的版本
            all_docs = {}
            for doc in doc_vector_results + doc_insights_results + doc_text_results:
                file_id = doc['file_id']
                # 智能去重策略：保留相似度更高的版本
                existing_score = all_docs.get(file_id, {}).get('similarity_score', 0)
                current_score = doc.get('similarity_score', doc.get('relevance_score', 0))
                
                if file_id not in all_docs or current_score > existing_score:
                    all_docs[file_id] = doc
            
            # 按相似度排序并限制数量
            relevant_docs = list(all_docs.values())
            relevant_docs.sort(key=lambda x: max(
                x.get('similarity_score', 0), 
                x.get('relevance_score', 0)
            ), reverse=True)
            result['relevant_documents'] = relevant_docs[:doc_top_k]
            
            # ============ 第二阶段：chunk级精确搜索 ============
            if relevant_docs:
                logger.info(f"在{len(relevant_docs)}个相关文档中搜索chunks...")
                
                # 提取文档ID，准备chunk搜索
                # 注意：file_id即为source_id，用于定位对应的chunk表和集合
                source_ids = []
                for doc in relevant_docs:
                    file_id = doc['file_id']
                    source_ids.append(file_id)
                
                # 策略1：chunk向量搜索
                # 在已筛选的文档内部，进行更精细的语义搜索
                chunk_vector_results = self.search_chunks_by_vector(
                    source_ids, query_vector, "summary_embedding", chunk_top_k
                )
                
                # 策略2：chunk全文搜索（条件性启用）
                # 如果有查询文本，在chunk内容中进行关键词搜索
                chunk_text_results = []
                if query_text:
                    chunk_text_results = self.search_chunks_by_text(
                        source_ids, query_text, top_k=chunk_top_k
                    )
                
                # ============ chunk结果合并与去重 ============
                # 同样使用字典去重，保留最高分的chunk版本
                all_chunk_results = {}
                for chunk in chunk_vector_results + chunk_text_results:
                    chunk_id = chunk['chunk_id']
                    
                    # 智能评分：优先考虑similarity_score，其次relevance_score
                    existing_score = max(
                        all_chunk_results.get(chunk_id, {}).get('similarity_score', 0),
                        all_chunk_results.get(chunk_id, {}).get('relevance_score', 0)
                    )
                    current_score = max(
                        chunk.get('similarity_score', 0),
                        chunk.get('relevance_score', 0)
                    )
                    
                    if chunk_id not in all_chunk_results or current_score > existing_score:
                        all_chunk_results[chunk_id] = chunk
                
                # 最终排序和截取
                # 使用多维评分：优先similarity_score，其次relevance_score
                final_chunks = list(all_chunk_results.values())
                final_chunks.sort(key=lambda x: max(
                    x.get('similarity_score', 0), 
                    x.get('relevance_score', 0)
                ), reverse=True)
                result['relevant_chunks'] = final_chunks[:final_top_k]
            
            # ============ 生成搜索报告 ============
            # 统计搜索结果和过程信息，便于调试和优化
            result['search_summary'] = {
                'total_documents_found': len(result['relevant_documents']),    # 第一阶段发现的文档数
                'total_chunks_found': len(result['relevant_chunks']),         # 第二阶段发现的chunk数
                'search_methods_used': ['vector_search'],                     # 使用的搜索方法列表
                'query_vector_dim': len(query_vector),                       # 查询向量维度
                'two_stage_search': True,                                     # 标识使用了两阶段搜索
                'document_sources': [doc['file_id'] for doc in result['relevant_documents']]  # 相关文档ID列表
            }
            
            # 如果使用了全文搜索，添加相关信息
            if query_text:
                result['search_summary']['search_methods_used'].append('fulltext_search')
                result['search_summary']['query_text'] = query_text
                result['search_summary']['hybrid_search'] = True  # 标识为混合搜索
            
            logger.info(f"综合搜索完成: {result['search_summary']}")
            return result
            
        except Exception as e:
            logger.error(f"综合搜索失败: {e}")
            # 即使出错也返回已有结果，提高系统容错性
            return result
    
    def close(self):
        """
        关闭数据库连接 - 资源清理
        
        【重要性】
        正确关闭数据库连接对系统稳定性至关重要：
        - 避免连接泄漏：未关闭的连接会占用服务器资源
        - 释放内存：清理本地缓存和连接对象
        - 优雅退出：确保没有未完成的事务
        
        【调用时机】
        - 应用程序关闭时
        - 长时间不使用检索器时
        - 出现连接错误需要重连时
        - 系统资源紧张时
        
        【容错设计】
        即使关闭过程出错，也不会影响应用程序的正常退出
        """
        try:
            # 关闭MySQL连接
            if self.mysql_conn:
                self.mysql_conn.close()
                self.mysql_conn = None
                
            # 断开Milvus连接
            if self.milvus_connected:
                connections.disconnect("default")
                self.milvus_connected = False
                
            logger.info("数据库连接已关闭")
        except Exception as e:
            # 关闭连接失败不应影响程序退出，只记录日志
            logger.error(f"关闭数据库连接失败: {e}")


def create_retriever(mysql_config: Dict = None, milvus_config: Dict = None) -> DBRetriever:
    """
    创建检索器实例的便捷函数
    ========================
    
    【设计目的】
    这是一个工厂函数，简化检索器的创建过程：
    - 提供默认配置，减少用户配置负担
    - 统一创建逻辑，便于维护和测试
    - 支持配置覆盖，保持灵活性
    
    【默认配置说明】
    MySQL默认配置：
    - host: 127.0.0.1（本地开发环境）
    - port: 3306（MySQL标准端口）
    - user: root（开发环境用户）
    - database: knowledge_rag（项目专用数据库）
    - charset: utf8mb4（支持emoji和特殊字符）
    
    Milvus默认配置：
    - host: 127.0.0.1（本地开发环境）
    - port: 19530（Milvus标准端口）
    - alias: default（默认连接别名）
    
    【使用示例】
    # 使用默认配置
    retriever = create_retriever()
    
    # 自定义MySQL配置
    mysql_cfg = {'host': 'prod-mysql.com', 'password': 'prod_pass'}
    retriever = create_retriever(mysql_config=mysql_cfg)
    
    Args:
        mysql_config: MySQL配置字典，None时使用默认配置
            - 只需提供需要覆盖的配置项
            - 会与默认配置合并
            
        milvus_config: Milvus配置字典，None时使用默认配置
            - 只需提供需要覆盖的配置项
            - 会与默认配置合并
        
    Returns:
        DBRetriever: 已初始化并连接到数据库的检索器实例
        
    Raises:
        Exception: 数据库连接失败时抛出异常
    """
    # 设置MySQL默认配置（适合开发环境）
    if mysql_config is None:
        mysql_config = {
            'host': '127.0.0.1',          # 本地MySQL服务器
            'port': 3306,                 # MySQL标准端口
            'user': 'root',               # 开发环境用户
            'password': 'devpass',        # 开发环境密码
            'database': 'knowledge_rag',  # 项目数据库名
            'charset': 'utf8mb4'          # 字符集（支持emoji）
        }
    
    # 设置Milvus默认配置（适合开发环境）
    if milvus_config is None:
        milvus_config = {
            'host': '127.0.0.1',    # 本地Milvus服务器
            'port': 19530,          # Milvus标准端口
            'alias': 'default'      # 连接别名
        }
    
    # 创建并返回检索器实例
    # 注意：构造函数会立即尝试连接数据库，失败会抛出异常
    return DBRetriever(mysql_config, milvus_config)
