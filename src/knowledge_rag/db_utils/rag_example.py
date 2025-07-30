"""
RAG系统两级数据库架构 - 完整使用示例

这个示例展示了如何使用两级RAG数据库架构：
1. 创建文档和chunk级数据库
2. 添加文档和chunks
3. 执行两级搜索
4. 管理数据

运行前确保：
1. db_server 服务已启动
2. 已创建并切换到实验环境

作者: XYZ-Algorithm-Team
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_rag.db_utils import (
    RAGDocumentManager, 
    RAGSearchEngine, 
    ChunkDataManager,
    RAGSchemaConfig
)

def example_1_setup_rag_system():
    """
    示例1：搭建RAG系统基础架构
    """
    print("=" * 60)
    print("示例1：搭建RAG系统基础架构")
    print("=" * 60)
    
    try:
        # 1. 初始化各个组件
        print("🔧 初始化RAG系统组件...")
        rag_manager = RAGDocumentManager()
        search_engine = RAGSearchEngine()
        chunk_manager = ChunkDataManager()
        schema_config = RAGSchemaConfig()
        
        print("✅ RAG系统组件初始化成功")
        print(f"   - 实验名称: {rag_manager.experiment_name}")
        print(f"   - MySQL数据库: {rag_manager.db_name}")
        print(f"   - 文档级向量集合: {search_engine.documents_collection_name}")
        
        return rag_manager, search_engine, chunk_manager, schema_config
        
    except Exception as e:
        print(f"❌ RAG系统初始化失败: {e}")
        return None, None, None, None

def example_2_add_documents(rag_manager, schema_config):
    """
    示例2：添加文档到RAG系统
    """
    print("\n" + "=" * 60)
    print("示例2：添加文档到RAG系统")
    print("=" * 60)
    
    if not rag_manager:
        print("❌ RAG系统未初始化，跳过此示例")
        return []
    
    # 示例文档数据
    documents_data = [
        {
            "title": "深度学习基础教程",
            "summary": "本教程介绍了深度学习的基本概念，包括神经网络、反向传播、卷积神经网络等核心技术。适合初学者系统性学习深度学习知识。",
            "keywords": ["深度学习", "神经网络", "卷积", "反向传播", "机器学习"],
            "document_embedding": np.random.rand(768).tolist(),  # 模拟文档级向量
            "summary_embedding": np.random.rand(768).tolist(),   # 模拟摘要向量
            "keywords_embedding": np.random.rand(768).tolist(),  # 模拟关键词向量
            "metadata": {
                "source": "educational",
                "difficulty": "beginner",
                "language": "zh-CN",
                "estimated_reading_time": 120
            }
        },
        {
            "title": "Transformer架构详解",
            "summary": "深入解析Transformer模型的工作原理，包括自注意力机制、位置编码、多头注意力等关键组件，以及在自然语言处理中的应用。",
            "keywords": ["Transformer", "注意力机制", "BERT", "GPT", "自然语言处理"],
            "document_embedding": np.random.rand(768).tolist(),
            "summary_embedding": np.random.rand(768).tolist(),
            "keywords_embedding": np.random.rand(768).tolist(),
            "metadata": {
                "source": "research_paper",
                "difficulty": "advanced",
                "language": "zh-CN",
                "citations": 15000
            }
        },
        {
            "title": "计算机视觉实战指南",
            "summary": "通过实际项目案例学习计算机视觉技术，涵盖图像分类、目标检测、语义分割等任务，提供完整的代码实现和解释。",
            "keywords": ["计算机视觉", "图像分类", "目标检测", "语义分割", "OpenCV"],
            "document_embedding": np.random.rand(768).tolist(),
            "summary_embedding": np.random.rand(768).tolist(),
            "keywords_embedding": np.random.rand(768).tolist(),
            "metadata": {
                "source": "practical_guide",
                "difficulty": "intermediate",
                "language": "zh-CN",
                "has_code": True
            }
        }
    ]
    
    doc_ids = []
    
    for i, doc_data in enumerate(documents_data):
        try:
            print(f"\n📄 添加文档 {i+1}/3: {doc_data['title']}")
            
            # 获取合适的schema配置
            chunk_schema = schema_config.get_chunk_schema("academic")
            
            # 添加文档
            doc_id = rag_manager.add_document(
                title=doc_data["title"],
                summary=doc_data["summary"],
                keywords=doc_data["keywords"],
                metadata=doc_data["metadata"],
                document_embedding=doc_data["document_embedding"],
                summary_embedding=doc_data["summary_embedding"],
                keywords_embedding=doc_data["keywords_embedding"],
                chunk_schema_config=chunk_schema
            )
            
            doc_ids.append(doc_id)
            print(f"   ✅ 文档添加成功，ID: {doc_id}")
            
            # 显示创建的chunk级数据库信息
            doc_info = rag_manager.get_document_info(doc_id)
            print(f"   📊 Chunk MySQL表: {doc_info['chunk_mysql_table']}")
            print(f"   🔍 Chunk Milvus集合: {doc_info['chunk_milvus_collection']}")
            
        except Exception as e:
            print(f"   ❌ 文档添加失败: {e}")
    
    print(f"\n✅ 文档添加完成，共添加 {len(doc_ids)} 个文档")
    print(f"📋 文档ID列表: {doc_ids}")
    
    return doc_ids

def example_3_add_chunks(chunk_manager, doc_ids):
    """
    示例3：向文档添加chunks
    """
    print("\n" + "=" * 60)
    print("示例3：向文档添加chunks")
    print("=" * 60)
    
    if not chunk_manager or not doc_ids:
        print("❌ 前置条件不满足，跳过此示例")
        return
    
    # 为每个文档添加示例chunks
    for doc_id in doc_ids:
        try:
            print(f"\n📝 为文档 {doc_id} 添加chunks...")
            
            # 获取文档信息
            doc_info = chunk_manager.rag_manager.get_document_info(doc_id)
            doc_title = doc_info['title']
            
            # 根据文档生成示例chunks
            if "深度学习" in doc_title:
                chunks_data = [
                    {
                        "chunk_text": "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的表示。深度学习的核心思想是通过组合简单的非线性变换来学习复杂的函数。",
                        "chunk_title": "第1章 深度学习概述",
                        "keywords": "深度学习,机器学习,神经网络,非线性变换",
                        "chunk_embedding": np.random.rand(768).tolist(),
                        "title_embedding": np.random.rand(768).tolist(),
                        "metadata": {"chapter": 1, "page": 10},
                        "token_count": 45
                    },
                    {
                        "chunk_text": "神经网络由多个神经元组成，每个神经元接收输入信号，通过激活函数产生输出。反向传播算法是训练神经网络的核心方法，通过计算梯度来更新网络参数。",
                        "chunk_title": "第2章 神经网络基础",
                        "keywords": "神经网络,神经元,激活函数,反向传播,梯度",
                        "chunk_embedding": np.random.rand(768).tolist(),
                        "title_embedding": np.random.rand(768).tolist(),
                        "metadata": {"chapter": 2, "page": 25},
                        "token_count": 52
                    },
                    {
                        "chunk_text": "卷积神经网络(CNN)特别适合处理图像数据。卷积层通过卷积操作提取局部特征，池化层减少参数数量，全连接层进行最终的分类或回归。",
                        "chunk_title": "第3章 卷积神经网络",
                        "keywords": "卷积神经网络,CNN,卷积层,池化层,图像处理",
                        "chunk_embedding": np.random.rand(768).tolist(),
                        "title_embedding": np.random.rand(768).tolist(),
                        "metadata": {"chapter": 3, "page": 40},
                        "token_count": 48
                    }
                ]
            elif "Transformer" in doc_title:
                chunks_data = [
                    {
                        "chunk_text": "Transformer模型完全基于注意力机制，摒弃了传统的循环和卷积结构。自注意力机制允许模型直接计算序列中任意两个位置之间的关系。",
                        "chunk_title": "Transformer架构概述",
                        "keywords": "Transformer,注意力机制,自注意力,序列模型",
                        "chunk_embedding": np.random.rand(768).tolist(),
                        "title_embedding": np.random.rand(768).tolist(),
                        "metadata": {"section": "introduction", "importance": 5},
                        "token_count": 55
                    },
                    {
                        "chunk_text": "多头注意力机制将输入分割成多个头，每个头独立计算注意力权重，然后将结果拼接。这种设计允许模型同时关注不同类型的信息。",
                        "chunk_title": "多头注意力机制",
                        "keywords": "多头注意力,注意力权重,并行计算,信息融合",
                        "chunk_embedding": np.random.rand(768).tolist(),
                        "title_embedding": np.random.rand(768).tolist(),
                        "metadata": {"section": "attention", "importance": 5},
                        "token_count": 47
                    }
                ]
            else:  # 计算机视觉
                chunks_data = [
                    {
                        "chunk_text": "图像分类是计算机视觉的基础任务，目标是将输入图像分配到预定义的类别中。常用的方法包括卷积神经网络、残差网络等。",
                        "chunk_title": "图像分类基础",
                        "keywords": "图像分类,卷积神经网络,残差网络,计算机视觉",
                        "chunk_embedding": np.random.rand(768).tolist(),
                        "title_embedding": np.random.rand(768).tolist(),
                        "metadata": {"task": "classification", "difficulty": 3},
                        "token_count": 42
                    },
                    {
                        "chunk_text": "目标检测不仅要识别图像中的对象类别，还要确定对象的位置。YOLO、R-CNN等算法是目标检测领域的重要突破。",
                        "chunk_title": "目标检测技术",
                        "keywords": "目标检测,YOLO,R-CNN,对象定位,边界框",
                        "chunk_embedding": np.random.rand(768).tolist(),
                        "title_embedding": np.random.rand(768).tolist(),
                        "metadata": {"task": "detection", "difficulty": 4},
                        "token_count": 38
                    }
                ]
            
            # 批量添加chunks
            chunk_ids = chunk_manager.add_chunks_to_document(doc_id, chunks_data)
            successful_chunks = [cid for cid in chunk_ids if cid is not None]
            
            print(f"   ✅ 成功添加 {len(successful_chunks)}/{len(chunks_data)} 个chunks")
            print(f"   📊 Chunk IDs: {successful_chunks}")
            
            # 获取统计信息
            stats = chunk_manager.get_document_chunk_statistics(doc_id)
            print(f"   📈 文档统计: 总chunks={stats.get('total_chunks', 0)}, 总tokens={stats.get('total_tokens', 0)}")
            
        except Exception as e:
            print(f"   ❌ 添加chunks失败: {e}")
    
    print("\n✅ Chunks添加完成")

def example_4_document_level_search(search_engine):
    """
    示例4：文档级搜索（第一级搜索）
    """
    print("\n" + "=" * 60)
    print("示例4：文档级搜索（第一级搜索）")
    print("=" * 60)
    
    if not search_engine:
        print("❌ 搜索引擎未初始化，跳过此示例")
        return []
    
    try:
        # 4.1 关键词搜索
        print("\n🔍 4.1 关键词搜索测试")
        keyword_results = search_engine.search_documents_by_keywords(
            keywords="深度学习 神经网络",
            search_fields=["title", "summary", "keywords"],
            top_k=5
        )
        
        print(f"   关键词搜索结果: {len(keyword_results)} 个文档")
        for i, doc in enumerate(keyword_results[:3]):
            print(f"   {i+1}. {doc['title']} (ID: {doc['id']})")
        
        # 4.2 向量搜索
        print("\n🎯 4.2 向量搜索测试")
        query_vector = np.random.rand(768).tolist()  # 模拟查询向量
        
        vector_results = search_engine.search_documents_by_vector(
            query_vector=query_vector,
            vector_field="document_embedding",
            similarity_threshold=0.3,  # 降低阈值以获得更多结果
            top_k=5
        )
        
        print(f"   向量搜索结果: {len(vector_results)} 个文档")
        for i, doc in enumerate(vector_results[:3]):
            similarity = doc.get('vector_search', {}).get('similarity', 0)
            print(f"   {i+1}. {doc['title']} (相似度: {similarity:.3f})")
        
        # 4.3 混合搜索
        print("\n🔀 4.3 混合搜索测试")
        hybrid_results = search_engine.search_documents(
            keywords="机器学习 算法",
            query_vector=query_vector,
            combine_results=True,
            keyword_weight=0.4,
            vector_weight=0.6,
            top_k=5
        )
        
        print(f"   混合搜索结果: {len(hybrid_results)} 个文档")
        for i, doc in enumerate(hybrid_results[:3]):
            combined_score = doc.get('combined_score', 0)
            print(f"   {i+1}. {doc['title']} (综合分数: {combined_score:.3f})")
        
        # 返回搜索到的文档ID，用于后续的chunk级搜索
        relevant_doc_ids = [doc['id'] for doc in hybrid_results]
        return relevant_doc_ids
        
    except Exception as e:
        print(f"❌ 文档级搜索失败: {e}")
        return []

def example_5_chunk_level_search(search_engine, doc_ids):
    """
    示例5：Chunk级搜索（第二级搜索）
    """
    print("\n" + "=" * 60)
    print("示例5：Chunk级搜索（第二级搜索）")
    print("=" * 60)
    
    if not search_engine or not doc_ids:
        print("❌ 前置条件不满足，跳过此示例")
        return
    
    try:
        # 5.1 单文档chunk搜索
        print(f"\n📄 5.1 单文档chunk搜索 (文档ID: {doc_ids[0]})")
        
        single_doc_chunks = search_engine.search_chunks_in_document(
            doc_id=doc_ids[0],
            keywords="神经网络 学习",
            query_vector=np.random.rand(768).tolist(),
            combine_results=True,
            top_k=3
        )
        
        print(f"   找到 {len(single_doc_chunks)} 个相关chunks:")
        for i, chunk in enumerate(single_doc_chunks):
            score = chunk.get('combined_score', chunk.get('vector_search', {}).get('similarity', 0))
            print(f"   {i+1}. {chunk.get('chunk_title', 'No Title')} (分数: {score:.3f})")
            print(f"      内容预览: {chunk['chunk_text'][:50]}...")
        
        # 5.2 并行chunk搜索
        print(f"\n⚡ 5.2 并行chunk搜索 (文档数: {len(doc_ids)})")
        
        parallel_results = search_engine.search_chunks_parallel(
            doc_ids=doc_ids,
            keywords="深度学习 算法",
            query_vector=np.random.rand(768).tolist(),
            top_k_per_doc=2,
            max_total_results=10
        )
        
        print(f"   搜索完成，{len(parallel_results)} 个文档有结果:")
        total_chunks = 0
        for doc_id, chunks in parallel_results.items():
            total_chunks += len(chunks)
            print(f"   📄 文档 {doc_id}: {len(chunks)} 个chunks")
            for i, chunk in enumerate(chunks[:2]):  # 只显示前2个
                score = chunk.get('combined_score', chunk.get('vector_search', {}).get('similarity', 0))
                print(f"      {i+1}. {chunk.get('chunk_title', 'No Title')} (分数: {score:.3f})")
        
        print(f"   总计找到 {total_chunks} 个相关chunks")
        
    except Exception as e:
        print(f"❌ Chunk级搜索失败: {e}")

def example_6_full_search_pipeline(search_engine):
    """
    示例6：完整的两级搜索流程
    """
    print("\n" + "=" * 60)
    print("示例6：完整的两级搜索流程")
    print("=" * 60)
    
    if not search_engine:
        print("❌ 搜索引擎未初始化，跳过此示例")
        return
    
    try:
        print("🚀 执行完整的两级RAG搜索...")
        
        # 模拟用户查询
        user_query = "如何使用神经网络进行图像识别"
        query_vector = np.random.rand(768).tolist()  # 实际应用中这里是query的embedding
        
        print(f"   用户查询: {user_query}")
        print("   正在执行两级搜索...")
        
        # 执行完整搜索
        results = search_engine.full_search(
            keywords=user_query,
            query_vector=query_vector,
            doc_top_k=5,
            chunk_top_k_per_doc=3,
            max_total_chunks=15,
            doc_similarity_threshold=0.3,
            chunk_similarity_threshold=0.3
        )
        
        # 显示搜索结果
        print(f"\n📊 搜索结果统计:")
        stats = results['search_stats']
        print(f"   - 相关文档数量: {stats['doc_count']}")
        print(f"   - 相关chunks数量: {stats['chunk_count']}")
        print(f"   - 有chunks的文档数: {stats['docs_with_chunks']}")
        
        print(f"\n📄 相关文档:")
        for i, doc in enumerate(results['relevant_documents'][:3]):
            score = doc.get('combined_score', doc.get('vector_search', {}).get('similarity', 0))
            print(f"   {i+1}. {doc['title']} (分数: {score:.3f})")
            print(f"      摘要: {doc['summary'][:80]}...")
        
        print(f"\n📝 相关chunks:")
        chunk_count = 0
        for doc_id, chunks in results['chunk_results'].items():
            doc_title = next((doc['title'] for doc in results['relevant_documents'] if doc['id'] == doc_id), f"文档{doc_id}")
            print(f"   📄 来自文档: {doc_title}")
            
            for i, chunk in enumerate(chunks[:2]):  # 每个文档只显示前2个chunks
                chunk_count += 1
                score = chunk.get('combined_score', chunk.get('vector_search', {}).get('similarity', 0))
                print(f"      {chunk_count}. {chunk.get('chunk_title', 'No Title')} (分数: {score:.3f})")
                print(f"         内容: {chunk['chunk_text'][:60]}...")
        
        print(f"\n✅ 两级搜索完成！为用户找到了最相关的信息")
        
    except Exception as e:
        print(f"❌ 完整搜索流程失败: {e}")

def example_7_data_management(rag_manager, chunk_manager):
    """
    示例7：数据管理操作
    """
    print("\n" + "=" * 60)
    print("示例7：数据管理操作")
    print("=" * 60)
    
    if not rag_manager:
        print("❌ 管理器未初始化，跳过此示例")
        return
    
    try:
        # 7.1 列出所有文档
        print("📋 7.1 文档列表")
        docs = rag_manager.list_documents(limit=10)
        print(f"   当前共有 {len(docs)} 个文档:")
        for doc in docs:
            status_emoji = {"pending": "⏳", "processing": "🔄", "completed": "✅", "failed": "❌"}
            emoji = status_emoji.get(doc['processing_status'], "❓")
            print(f"   {emoji} ID: {doc['id']}, 标题: {doc['title']}, Chunks: {doc['chunk_count']}")
        
        if docs:
            # 7.2 查看文档详细信息
            doc_id = docs[0]['id']
            print(f"\n🔍 7.2 文档详细信息 (ID: {doc_id})")
            doc_info = rag_manager.get_document_info(doc_id)
            
            print(f"   标题: {doc_info['title']}")
            print(f"   摘要: {doc_info['summary'][:100]}...")
            print(f"   关键词: {', '.join(doc_info.get('keywords', []))}")
            print(f"   Chunk MySQL表: {doc_info['chunk_mysql_table']}")
            print(f"   Chunk Milvus集合: {doc_info['chunk_milvus_collection']}")
            print(f"   处理状态: {doc_info['processing_status']}")
            print(f"   Chunk数量: {doc_info['chunk_count']}")
            
            # 7.3 查看文档的chunks
            if chunk_manager and doc_info['chunk_count'] > 0:
                print(f"\n📝 7.3 文档chunks (ID: {doc_id})")
                chunks = chunk_manager.get_document_chunks(doc_id, limit=5)
                
                print(f"   找到 {len(chunks)} 个chunks:")
                for chunk in chunks:
                    print(f"   - Chunk {chunk['chunk_index']}: {chunk.get('chunk_title', 'No Title')}")
                    print(f"     内容: {chunk['chunk_text'][:80]}...")
                    print(f"     Tokens: {chunk.get('token_count', 'N/A')}")
                
                # 7.4 chunk统计信息
                print(f"\n📊 7.4 Chunk统计信息")
                stats = chunk_manager.get_document_chunk_statistics(doc_id)
                print(f"   总chunk数: {stats.get('total_chunks', 0)}")
                print(f"   总token数: {stats.get('total_tokens', 0)}")
                print(f"   平均token数: {stats.get('avg_token_count', 0):.1f}")
                print(f"   最大token数: {stats.get('max_token_count', 0)}")
            
            # 7.5 更新文档状态
            print(f"\n🔄 7.5 更新文档状态")
            current_status = doc_info['processing_status']
            new_status = "completed" if current_status != "completed" else "processing"
            
            success = rag_manager.update_document_status(doc_id, new_status)
            if success:
                print(f"   ✅ 文档状态已更新: {current_status} -> {new_status}")
            else:
                print(f"   ❌ 文档状态更新失败")
        
    except Exception as e:
        print(f"❌ 数据管理操作失败: {e}")

def main():
    """主函数 - 运行所有示例"""
    print("🚀 RAG系统两级数据库架构 - 完整示例")
    print("=" * 80)
    
    # 示例1：搭建RAG系统
    rag_manager, search_engine, chunk_manager, schema_config = example_1_setup_rag_system()
    
    if not all([rag_manager, search_engine, chunk_manager, schema_config]):
        print("❌ 系统初始化失败，无法继续运行示例")
        return
    
    # 示例2：添加文档
    doc_ids = example_2_add_documents(rag_manager, schema_config)
    
    # 示例3：添加chunks
    example_3_add_chunks(chunk_manager, doc_ids)
    
    # 示例4：文档级搜索
    relevant_doc_ids = example_4_document_level_search(search_engine)
    
    # 示例5：Chunk级搜索
    example_5_chunk_level_search(search_engine, relevant_doc_ids or doc_ids)
    
    # 示例6：完整搜索流程
    example_6_full_search_pipeline(search_engine)
    
    # 示例7：数据管理
    example_7_data_management(rag_manager, chunk_manager)
    
    print("\n" + "=" * 80)
    print("✅ 所有示例运行完成！")
    print("\n💡 总结:")
    print("   1. ✅ 成功搭建了两级RAG数据库架构")
    print("   2. ✅ 实现了文档级和chunk级的数据管理")
    print("   3. ✅ 演示了完整的两级搜索流程")
    print("   4. ✅ 展示了灵活的数据操作功能")
    print("\n🎯 你现在可以基于这个架构构建自己的RAG系统了！")

if __name__ == "__main__":
    main()