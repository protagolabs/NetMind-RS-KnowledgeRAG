#!/usr/bin/env python3
"""
KnowledgeRAG 快速入门示例
演示如何创建一个完整的RAG实验流程

使用方法:
1. 确保服务已启动: docker compose up -d
2. 运行示例: python quick_start_example.py
"""

import os
import sys
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 添加源代码路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

# 导入工具模块
from knowledge_rag.utils.s3_local import S3LocalClient
from knowledge_rag.utils.mysql_client import get_mysql_client, ChunkIn
from knowledge_rag.utils.milvus_client import get_milvus_client
from knowledge_rag.utils.flexible_search import FlexibleSearchEngine, SearchQuery

class QuickStartRAG:
    """快速入门RAG示例"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.s3_client = S3LocalClient(base_path="./data/local_object_store")
        
        # 初始化数据库客户端
        try:
            self.mysql_client = get_mysql_client()
            print("✅ MySQL 客户端初始化成功")
        except Exception as e:
            print(f"❌ MySQL 客户端初始化失败: {e}")
            self.mysql_client = None
        
        try:
            # 添加超时处理
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Milvus连接超时")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)  # 5秒超时
            
            self.milvus_client = get_milvus_client()
            signal.alarm(0)  # 取消超时
            print("✅ Milvus 客户端初始化成功")
        except Exception as e:
            signal.alarm(0)  # 确保取消超时
            print(f"❌ Milvus 客户端初始化失败: {e}")
            self.milvus_client = None
        
        try:
            self.search_engine = FlexibleSearchEngine(experiment_name)
            print("✅ FlexibleSearchEngine 初始化成功")
        except Exception as e:
            print(f"❌ FlexibleSearchEngine 初始化失败: {e}")
            self.search_engine = None
    
    def create_sample_documents(self) -> List[Dict]:
        """创建示例文档"""
        sample_docs = [
            {
                "title": "机器学习基础",
                "content": """
                机器学习是人工智能的一个重要分支，它让计算机能够从数据中学习，
                而无需明确编程。主要包括监督学习、无监督学习和强化学习三种类型。
                
                监督学习使用标记的训练数据来学习输入和输出之间的映射关系。
                常见的监督学习算法包括线性回归、逻辑回归、决策树、随机森林等。
                
                无监督学习处理没有标记的数据，目标是发现数据中的隐藏模式。
                主要方法包括聚类、降维、关联规则挖掘等。
                """,
                "category": "AI基础"
            },
            {
                "title": "深度学习概述",
                "content": """
                深度学习是机器学习的一个子领域，基于人工神经网络。
                它能够自动学习数据的多层表示，在图像识别、自然语言处理等领域取得了突破性进展。
                
                深度学习的核心是多层神经网络，包括全连接层、卷积层、循环层等。
                常见的深度学习架构包括CNN（卷积神经网络）、RNN（循环神经网络）、
                Transformer等。
                
                深度学习在计算机视觉、语音识别、机器翻译等任务上表现出色。
                """,
                "category": "深度学习"
            },
            {
                "title": "RAG系统设计",
                "content": """
                RAG（检索增强生成）是一种结合了检索和生成的AI架构。
                它首先从知识库中检索相关信息，然后基于检索到的信息生成回答。
                
                RAG系统的核心组件包括：
                1. 文档存储：存储原始知识文档
                2. 向量化：将文档转换为向量表示
                3. 检索系统：基于查询找到相关文档
                4. 生成模型：基于检索结果生成最终回答
                
                RAG系统在问答、文档总结、内容创作等任务中表现优异。
                """,
                "category": "RAG技术"
            }
        ]
        
        return sample_docs
    
    def upload_document(self, title: str, content: str, category: str, user_id: int = 1) -> tuple:
        """上传文档到系统"""
        print(f"\n📄 上传文档: {title}")
        
        # 1. 生成文档UUID
        doc_uuid = str(uuid.uuid4())
        
        # 2. 创建临时文件
        temp_file = Path(f"temp_{doc_uuid}.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 3. 上传到对象存储
        with open(temp_file, 'rb') as file:
            uri = self.s3_client.put_object(
                user_id=user_id,
                doc_uuid=doc_uuid,
                version_label="v1.0",
                filename=f"{title}.txt",
                file_stream=file,
                content_type="text/plain",
                metadata={"category": category}
            )
        
        # 4. 清理临时文件
        temp_file.unlink()
        
        # 5. 存储文档元数据到MySQL（模拟）
        if self.mysql_client:
            try:
                doc_id = self.mysql_client.create_document(
                    user_id=user_id,
                    title=title,
                    mime_type="text/plain"
                )
                
                import hashlib
                # 生成唯一的checksum
                checksum = hashlib.md5((uri + str(datetime.now())).encode()).hexdigest()
                
                version_id = self.mysql_client.create_version(
                    doc_id=doc_id,
                    source_uri=uri,
                    version_label="v1.0",
                    checksum=checksum
                )
                
                print(f"   ✅ 文档上传成功: doc_id={doc_id}, version_id={version_id}")
                return doc_id, version_id, doc_uuid
                
            except Exception as e:
                print(f"   ❌ 数据库操作失败: {e}")
                return None, None, doc_uuid
        
        return None, None, doc_uuid
    
    def process_text_chunks(self, content: str, doc_uuid: str, version_id: int = None) -> List[ChunkIn]:
        """处理文本分块"""
        print(f"   📝 处理文本分块...")
        
        # 简单分块策略：按段落分割
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunk_records = []
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 50:  # 过滤太短的段落
                chunk_uid = str(uuid.uuid4())
                chunk_record = ChunkIn(
                    seq_no=i,
                    chunk_uid=chunk_uid,
                    text=paragraph,
                    token_count=len(paragraph.split())
                )
                chunk_records.append(chunk_record)
        
        # 存储chunks到MySQL（模拟）
        if self.mysql_client and version_id:
            try:
                self.mysql_client.create_chunks(version_id, chunk_records)
                print(f"   ✅ 创建 {len(chunk_records)} 个文本块")
            except Exception as e:
                print(f"   ❌ 文本块存储失败: {e}")
        
        return chunk_records
    
    def generate_mock_embeddings(self, chunks: List[ChunkIn], doc_uuid: str) -> bool:
        """生成模拟embeddings"""
        print(f"   🔢 生成向量embeddings...")
        
        if not self.milvus_client:
            print("   ⚠️  Milvus 客户端未初始化，跳过向量存储")
            return False
        
        try:
            for chunk in chunks:
                # 生成模拟向量（实际应用中使用真实的embedding模型）
                import numpy as np
                # 从配置获取向量维度
                from knowledge_rag.config import get_embedding_settings
                embedding_settings = get_embedding_settings()
                mock_vector = np.random.rand(embedding_settings.dimension).tolist()
                
                # 存储到Milvus
                success = self.milvus_client.upsert_embedding(
                    embedding_id=hash(chunk.chunk_uid) % (2**63),  # 确保为正数
                    user_id=1,
                    doc_uuid=doc_uuid,
                    version_label="v1.0",
                    chunk_uid=chunk.chunk_uid,
                    vector=mock_vector
                )
                
                if not success:
                    print(f"   ❌ 向量存储失败: {chunk.chunk_uid}")
                    return False
            
            print(f"   ✅ 向量存储成功: {len(chunks)} 个向量")
            return True
            
        except Exception as e:
            print(f"   ❌ 向量处理失败: {e}")
            return False
    
    def test_search(self, questions: List[str]):
        """测试搜索功能"""
        print(f"\n🔍 测试搜索功能...")
        
        if not self.search_engine:
            print("   ⚠️  搜索引擎未初始化，跳过搜索测试")
            return
        
        for i, question in enumerate(questions, 1):
            print(f"\n   📋 测试 {i}: {question}")
            
            try:
                # 创建搜索查询
                query = SearchQuery(
                    query_text=question,
                    query_type="semantic",
                    top_k=3,
                    threshold=0.5,
                    experiment_name=self.experiment_name
                )
                
                # 执行搜索
                results = self.search_engine.search(query)
                
                # 显示结果
                if results:
                    for j, result in enumerate(results, 1):
                        print(f"      {j}. 得分: {result.score:.3f}")
                        print(f"         内容: {result.content[:100]}...")
                        print(f"         来源: {result.source}")
                else:
                    print("      📭 未找到相关结果")
                    
            except Exception as e:
                print(f"      ❌ 搜索失败: {e}")
    
    def run_complete_demo(self):
        """运行完整演示"""
        print("🚀 开始 KnowledgeRAG 快速入门演示")
        print("=" * 50)
        
        # 1. 创建示例文档
        sample_docs = self.create_sample_documents()
        
        # 2. 上传和处理文档
        print("\n📂 阶段 1: 文档上传与处理")
        for doc in sample_docs:
            doc_id, version_id, doc_uuid = self.upload_document(
                doc["title"], 
                doc["content"], 
                doc["category"]
            )
            
            if doc_uuid:
                # 处理文本分块
                chunks = self.process_text_chunks(doc["content"], doc_uuid, version_id)
                
                # 生成embeddings
                self.generate_mock_embeddings(chunks, doc_uuid)
        
        # 3. 测试搜索
        print("\n🔍 阶段 2: 搜索功能测试")
        test_questions = [
            "什么是机器学习？",
            "深度学习的主要特点是什么？",
            "RAG系统包含哪些组件？",
            "如何进行无监督学习？"
        ]
        
        self.test_search(test_questions)
        
        # 4. 总结
        print("\n✅ 演示完成!")
        print("=" * 50)
        print("📊 演示总结:")
        print("   - 上传了 3 个示例文档")
        print("   - 处理了文本分块和向量化")
        print("   - 测试了 4 个搜索查询")
        print("   - 展示了完整的 RAG 流程")
        
        print("\n🎯 下一步可以:")
        print("   1. 查看实验管理功能: python experiment_manager.py --interactive")
        print("   2. 检查健康状态: python experiment_data.py --action health-check")
        print("   3. 上传自己的文档进行测试")
        print("   4. 阅读详细指南: RAG_EXPERIMENT_GUIDE.md")

def main():
    """主函数"""
    print("🎯 KnowledgeRAG 快速入门示例")
    print("请确保已启动服务: docker compose up -d")
    
    # 创建实验演示
    demo = QuickStartRAG("quick_start_demo")
    
    # 运行完整演示
    demo.run_complete_demo()

if __name__ == "__main__":
    main() 