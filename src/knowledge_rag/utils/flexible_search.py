"""
KnowledgeRAG 灵活搜索工具
作者: XYZ-Algorithm-Team
用途: 支持动态查询不同表结构，适合实验环境
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .mysql_client import get_mysql_client
from .milvus_client import get_milvus_client
from .logging_utils import get_logger, LogLevel, LogCategory

logger = logging.getLogger(__name__)

@dataclass
class SearchQuery:
    """搜索查询数据类"""
    query_text: str
    query_type: str = "semantic"  # semantic, keyword, hybrid, custom
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 10
    threshold: float = 0.7
    experiment_name: Optional[str] = None
    table_mapping: Optional[Dict[str, str]] = None  # 自定义表映射

@dataclass
class SearchResult:
    """搜索结果数据类"""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    source: str  # 来源表

class FlexibleSearchEngine:
    """灵活搜索引擎"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name
        self.mysql_client = get_mysql_client()
        self.milvus_client = None
        self.logger = get_logger()
        
        # 尝试连接Milvus
        try:
            self.milvus_client = get_milvus_client()
        except Exception as e:
            logger.warning(f"Milvus连接失败，将使用纯文本搜索: {e}")
        
        # 默认表映射
        self.default_table_mapping = {
            'users': 'users',
            'documents': 'documents', 
            'chunks': 'chunks',
            'vectors': 'embeddings'
        }
    
    def set_experiment(self, experiment_name: str):
        """设置实验环境"""
        self.experiment_name = experiment_name
        # 更新数据库连接
        import os
        os.environ['MYSQL_DB'] = f"knowledge_rag_{experiment_name}"
        # 重新获取客户端
        self.mysql_client = get_mysql_client()
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """执行搜索"""
        try:
            # 选择搜索策略
            if query.query_type == "semantic":
                return self._semantic_search(query)
            elif query.query_type == "keyword":
                return self._keyword_search(query)
            elif query.query_type == "hybrid":
                return self._hybrid_search(query)
            elif query.query_type == "custom":
                return self._custom_search(query)
            else:
                raise ValueError(f"不支持的查询类型: {query.query_type}")
        
        except Exception as e:
            self.logger.log_error(
                LogCategory.RETRIEVAL,
                f"搜索失败: {query.query_type}",
                e,
                {"query": query.query_text, "experiment": self.experiment_name}
            )
            raise
    
    def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """语义搜索"""
        if not self.milvus_client:
            logger.warning("Milvus不可用，回退到关键词搜索")
            return self._keyword_search(query)
        
        results = []
        
        # 1. 向量搜索
        try:
            # 这里需要embedding模型，先用模拟数据
            query_vector = self._get_query_embedding(query.query_text)
            
            # 搜索向量
            vector_results = self.milvus_client.search(
                query_vector=query_vector,
                top_k=query.top_k,
                user_id=query.filters.get('user_id') if query.filters else None
            )
            
            # 获取对应的文本内容
            for result in vector_results:
                content = self._get_content_by_chunk_id(result.chunk_uid)
                if content:
                    results.append(SearchResult(
                        id=result.chunk_uid,
                        score=1.0 - result.distance,  # 转换为相似度
                        content=content,
                        metadata={
                            'doc_uuid': result.doc_uuid,
                            'version_label': result.version_label,
                            'timestamp': result.timestamp
                        },
                        source='milvus'
                    ))
        
        except Exception as e:
            logger.warning(f"向量搜索失败: {e}")
        
        return results
    
    def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """关键词搜索"""
        results = []
        
        try:
            # 获取表映射
            table_mapping = query.table_mapping or self.default_table_mapping
            
            # 检查表是否存在
            tables = self.mysql_client.list_tables()
            available_tables = [t for t in table_mapping.values() if t in tables]
            
            if not available_tables:
                logger.warning(f"没有找到可搜索的表: {available_tables}")
                return results
            
            # 在每个表中搜索
            for table_name in available_tables:
                table_results = self._search_in_table(
                    table_name, 
                    query.query_text, 
                    query.filters,
                    query.top_k
                )
                results.extend(table_results)
        
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
        
        # 按相关性排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:query.top_k]
    
    def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """混合搜索"""
        # 分别执行语义搜索和关键词搜索
        semantic_results = self._semantic_search(query)
        keyword_results = self._keyword_search(query)
        
        # 合并结果
        combined_results = {}
        
        # 添加语义搜索结果（权重0.6）
        for result in semantic_results:
            combined_results[result.id] = result
            result.score *= 0.6
        
        # 添加关键词搜索结果（权重0.4）
        for result in keyword_results:
            if result.id in combined_results:
                # 合并分数
                combined_results[result.id].score += result.score * 0.4
            else:
                result.score *= 0.4
                combined_results[result.id] = result
        
        # 排序并返回
        results = list(combined_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:query.top_k]
    
    def _custom_search(self, query: SearchQuery) -> List[SearchResult]:
        """自定义搜索"""
        if not query.filters or 'custom_sql' not in query.filters:
            raise ValueError("自定义搜索需要提供custom_sql")
        
        results = []
        
        try:
            with self.mysql_client.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # 执行自定义SQL
                custom_sql = query.filters['custom_sql']
                cursor.execute(custom_sql)
                rows = cursor.fetchall()
                
                # 转换为搜索结果
                for i, row in enumerate(rows):
                    # 尝试自动检测内容字段
                    content = self._extract_content_from_row(row)
                    
                    results.append(SearchResult(
                        id=str(row.get('id', i)),
                        score=1.0,  # 自定义搜索不计算相关性
                        content=content,
                        metadata=row,
                        source='custom_sql'
                    ))
                
                cursor.close()
        
        except Exception as e:
            logger.error(f"自定义搜索失败: {e}")
        
        return results
    
    def _search_in_table(self, table_name: str, query_text: str, 
                        filters: Optional[Dict], top_k: int) -> List[SearchResult]:
        """在指定表中搜索"""
        results = []
        
        try:
            with self.mysql_client.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # 获取表结构
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                
                # 找到文本字段
                text_columns = []
                for col in columns:
                    col_type = col['Type'].lower()
                    if any(t in col_type for t in ['text', 'varchar', 'char']):
                        text_columns.append(col['Field'])
                
                if not text_columns:
                    return results
                
                # 构建搜索SQL
                search_conditions = []
                for col in text_columns:
                    search_conditions.append(f"{col} LIKE %s")
                
                where_clause = " OR ".join(search_conditions)
                
                # 添加过滤条件
                if filters:
                    for key, value in filters.items():
                        if key != 'user_id':  # user_id特殊处理
                            where_clause += f" AND {key} = %s"
                
                sql = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT %s"
                
                # 准备参数
                params = [f"%{query_text}%"] * len(text_columns)
                if filters:
                    for key, value in filters.items():
                        if key != 'user_id':
                            params.append(value)
                params.append(top_k)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                # 转换为搜索结果
                for row in rows:
                    content = self._extract_content_from_row(row)
                    score = self._calculate_text_similarity(query_text, content)
                    
                    results.append(SearchResult(
                        id=str(row.get('id', '')),
                        score=score,
                        content=content,
                        metadata=row,
                        source=table_name
                    ))
                
                cursor.close()
        
        except Exception as e:
            logger.error(f"表搜索失败 {table_name}: {e}")
        
        return results
    
    def _get_query_embedding(self, query_text: str) -> List[float]:
        """获取查询向量（模拟）"""
        raise NotImplementedError("Not implemented, please implement this function in your own code.")
        # 这里应该调用真实的embedding模型
        # 现在返回随机向量作为示例，维度与配置一致
        from ..config import get_embedding_settings
        embedding_settings = get_embedding_settings()
        return np.random.random(embedding_settings.dimension).tolist()
    
    def _get_content_by_chunk_id(self, chunk_uid: str) -> Optional[str]:
        """根据chunk_uid获取内容"""
        try:
            with self.mysql_client.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # 尝试从chunks表获取
                cursor.execute("SELECT text FROM chunks WHERE chunk_uid = %s", (chunk_uid,))
                row = cursor.fetchone()
                
                if row:
                    return row['text']
                
                cursor.close()
        except Exception as e:
            logger.error(f"获取chunk内容失败: {e}")
        
        return None
    
    def _extract_content_from_row(self, row: Dict) -> str:
        """从行数据中提取内容"""
        # 优先级：text > content > title > name
        for field in ['text', 'content', 'title', 'name']:
            if field in row and row[field]:
                return str(row[field])
        
        # 如果没有明显的内容字段，返回所有字段的拼接
        return " ".join(str(v) for v in row.values() if v)
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """计算文本相似度（简单实现）"""
        if not query or not text:
            return 0.0
        
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words)
    
    def explain_search(self, query: SearchQuery) -> Dict[str, Any]:
        """解释搜索策略"""
        explanation = {
            'query_type': query.query_type,
            'experiment': self.experiment_name,
            'available_tables': self.mysql_client.list_tables(),
            'milvus_available': self.milvus_client is not None,
            'strategy': ''
        }
        
        if query.query_type == "semantic":
            if self.milvus_client:
                explanation['strategy'] = 'Vector search in Milvus + MySQL content lookup'
            else:
                explanation['strategy'] = 'Fallback to keyword search (Milvus unavailable)'
        elif query.query_type == "keyword":
            explanation['strategy'] = 'Full-text search across all text columns'
        elif query.query_type == "hybrid":
            explanation['strategy'] = 'Semantic (60%) + Keyword (40%) combined search'
        elif query.query_type == "custom":
            explanation['strategy'] = 'Custom SQL execution'
        
        return explanation

class ExperimentSearchAnalyzer:
    """实验搜索分析器"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.search_engine = FlexibleSearchEngine(experiment_name)
        self.logger = get_logger()
    
    def analyze_search_performance(self, queries: List[str], 
                                  query_types: List[str] = None) -> Dict[str, Any]:
        """分析搜索性能"""
        if query_types is None:
            query_types = ["semantic", "keyword", "hybrid"]
        
        results = {}
        
        for query_type in query_types:
            type_results = {
                'total_queries': 0,
                'total_time': 0,
                'avg_time': 0,
                'avg_results': 0,
                'errors': 0
            }
            
            for query_text in queries:
                try:
                    start_time = datetime.now()
                    
                    search_query = SearchQuery(
                        query_text=query_text,
                        query_type=query_type,
                        top_k=10
                    )
                    
                    search_results = self.search_engine.search(search_query)
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds() * 1000
                    
                    type_results['total_queries'] += 1
                    type_results['total_time'] += duration
                    type_results['avg_results'] += len(search_results)
                    
                except Exception as e:
                    type_results['errors'] += 1
                    logger.error(f"搜索分析失败: {e}")
            
            if type_results['total_queries'] > 0:
                type_results['avg_time'] = type_results['total_time'] / type_results['total_queries']
                type_results['avg_results'] = type_results['avg_results'] / type_results['total_queries']
            
            results[query_type] = type_results
        
        return results
    
    def generate_search_report(self, analysis_results: Dict[str, Any]) -> str:
        """生成搜索报告"""
        report = f"实验搜索性能报告 - {self.experiment_name}\n"
        report += "=" * 50 + "\n\n"
        
        for query_type, results in analysis_results.items():
            report += f"搜索类型: {query_type}\n"
            report += f"  查询总数: {results['total_queries']}\n"
            report += f"  平均耗时: {results['avg_time']:.2f}ms\n"
            report += f"  平均结果数: {results['avg_results']:.1f}\n"
            report += f"  错误次数: {results['errors']}\n"
            report += "\n"
        
        return report

def main():
    """命令行测试工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description='灵活搜索工具测试')
    parser.add_argument('--experiment', '-e', help='实验名称')
    parser.add_argument('--query', '-q', required=True, help='搜索查询')
    parser.add_argument('--type', '-t', choices=['semantic', 'keyword', 'hybrid', 'custom'], 
                       default='keyword', help='搜索类型')
    parser.add_argument('--top-k', '-k', type=int, default=10, help='返回结果数')
    parser.add_argument('--explain', action='store_true', help='解释搜索策略')
    
    args = parser.parse_args()
    
    # 创建搜索引擎
    search_engine = FlexibleSearchEngine(args.experiment)
    
    # 创建搜索查询
    search_query = SearchQuery(
        query_text=args.query,
        query_type=args.type,
        top_k=args.top_k
    )
    
    # 解释搜索策略
    if args.explain:
        explanation = search_engine.explain_search(search_query)
        print(json.dumps(explanation, indent=2, ensure_ascii=False))
        return
    
    # 执行搜索
    try:
        start_time = datetime.now()
        results = search_engine.search(search_query)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds() * 1000
        
        print(f"搜索完成: {len(results)} 个结果, 耗时: {duration:.2f}ms")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [分数: {result.score:.3f}] [来源: {result.source}]")
            print(f"   内容: {result.content[:200]}...")
            print(f"   元数据: {result.metadata}")
            print()
    
    except Exception as e:
        print(f"搜索失败: {e}")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main()) 