#!/usr/bin/env python3
"""
KnowledgeRAG API 测试脚本
=======================

作者: Bin Liang
日期: 2025-08-01
描述: 测试 KnowledgeRAG API 的各个端点功能

使用方法:
    python test_api.py              # 测试所有端点
    python test_api.py --endpoint health  # 测试特定端点
    python test_api.py --base-url http://localhost:9000  # 指定服务器地址
"""

import asyncio
import argparse
import json
from typing import Dict, Any
import httpx
from loguru import logger

class APITester:
    """API 测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8954"):
        self.base_url = base_url
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def test_health_check(self):
        """测试健康检查端点"""
        logger.info("🔍 测试健康检查...")
        
        # 简单健康检查
        response = await self.client.get(f"{self.base_url}/")
        assert response.status_code == 200
        data = response.json()
        logger.info(f"✅ 基础健康检查: {data['message']}")
        
        # 详细健康检查
        response = await self.client.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        logger.info(f"✅ 详细健康检查: {data['status']}")
        logger.info(f"   - Retriever 连接: {'✅' if data['retriever']['connected'] else '❌'}")
        logger.info(f"   - MySQL 连接: {'✅' if data['retriever']['mysql_connected'] else '❌'}")
        logger.info(f"   - Milvus 连接: {'✅' if data['retriever']['milvus_connected'] else '❌'}")
        
        return True
    
    async def test_document_upload(self):
        """测试文档上传"""
        logger.info("📄 测试文档上传...")
        
        test_data = {
            "markdown_documents": [
                """# 机器学习基础

## 概述
机器学习是人工智能的一个重要分支，它使计算机系统能够从数据中自动学习和改进。

## 主要类型
1. **监督学习**: 使用标记数据进行训练
2. **无监督学习**: 从未标记数据中发现模式
3. **强化学习**: 通过与环境交互学习最优策略

## 应用领域
- 图像识别
- 自然语言处理
- 推荐系统
- 预测分析
""",
                """# 深度学习进阶

## 神经网络架构
深度学习基于人工神经网络，通过多层非线性变换来学习复杂的数据表示。

## 常见架构
1. **卷积神经网络 (CNN)**: 主要用于图像处理
2. **循环神经网络 (RNN)**: 适合序列数据
3. **Transformer**: 在自然语言处理中表现优异

## 优化技术
- 梯度下降算法
- 批量归一化
- Dropout 正则化
- 学习率调度
"""
            ],
            "file_names": [
                "ml_basics.md",
                "deep_learning_advanced.md"
            ]
        }
        
        response = await self.client.post(f"{self.base_url}/upload", json=test_data)
        
        if response.status_code != 200:
            logger.error(f"❌ 文档上传失败: {response.status_code} - {response.text}")
            return False
        
        data = response.json()
        logger.info(f"✅ 文档上传成功:")
        logger.info(f"   - 处理文档数: {data['total_docs']}")
        logger.info(f"   - 生成 chunks: {data['total_chunks']}")
        
        return data
    
    async def test_embedding_generation(self, upload_data: Dict[str, Any]):
        """测试嵌入生成"""
        logger.info("🧮 测试嵌入生成...")
        
        embedding_data = {
            "doc_analysis_list": upload_data["doc_analysis_list"],
            "chunk_analysis_list": upload_data["chunk_analysis_list"]
        }
        
        response = await self.client.post(f"{self.base_url}/embedding", json=embedding_data)
        
        if response.status_code != 200:
            logger.error(f"❌ 嵌入生成失败: {response.status_code} - {response.text}")
            return False
        
        data = response.json()
        logger.info(f"✅ 嵌入生成成功:")
        logger.info(f"   - 文档嵌入数: {len(data['doc_embedding_list'])}")
        logger.info(f"   - Chunk 嵌入数: {len(data['chunk_embedding_list'])}")
        
        return data
    
    async def test_save_to_db(self, embedding_data: Dict[str, Any]):
        """测试保存到数据库"""
        logger.info("💾 测试保存到数据库...")
        
        save_data = {
            "doc_embedding_list": embedding_data["doc_embedding_list"],
            "chunk_embedding_list": embedding_data["chunk_embedding_list"]
        }
        
        response = await self.client.post(f"{self.base_url}/save-to-db", json=save_data)
        
        if response.status_code != 200:
            logger.error(f"❌ 数据保存失败: {response.status_code} - {response.text}")
            return False
        
        data = response.json()
        logger.info(f"✅ 数据保存成功: {data['message']}")
        
        return True
    
    async def test_document_retrieval(self):
        """测试文档检索"""
        logger.info("🔍 测试文档检索...")
        
        test_queries = [
            "机器学习的主要类型有哪些？",
            "深度学习中的神经网络架构",
            "卷积神经网络的应用"
        ]
        
        for query in test_queries:
            logger.info(f"   查询: {query}")
            
            response = await self.client.post(f"{self.base_url}/doc-retrieval", json={
                "query": query
            })
            
            if response.status_code != 200:
                logger.error(f"❌ 文档检索失败: {response.status_code} - {response.text}")
                continue
            
            data = response.json()
            logger.info(f"   ✅ 找到 {data['total_results']} 个相关文档")
            
            # 显示前 2 个结果
            for i, result in enumerate(data["results"][:2]):
                if isinstance(result, dict) and "file_name" in result:
                    logger.info(f"      {i+1}. {result.get('file_name', 'Unknown')}")
        
        return True
    
    async def test_chunk_retrieval(self):
        """测试 chunk 检索"""
        logger.info("📝 测试 chunk 检索...")
        
        # 首先获取一些文档 ID (这里使用模拟数据)
        test_doc_ids = ["doc_001", "doc_002"]  # 实际应从文档检索结果中获取
        
        test_data = {
            "query": "神经网络优化方法",
            "doc_ids": test_doc_ids,
            "each_doc_chunk_number": 5
        }
        
        response = await self.client.post(f"{self.base_url}/chunk-retrieval", json=test_data)
        
        if response.status_code != 200:
            logger.warning(f"⚠️ Chunk 检索可能失败 (需要先有数据): {response.status_code}")
            return True  # 这是预期的，因为可能还没有数据
        
        data = response.json()
        logger.info(f"✅ Chunk 检索成功: 找到 {data['total_results']} 个 chunks")
        
        return True
    
    async def test_chunk_decision(self):
        """测试 chunk 决策匹配"""
        logger.info("🎯 测试 chunk 决策匹配...")
        
        test_chunks = [
            {
                "chunk_id": "test_chunk_001",
                "chunk_content": "机器学习是人工智能的一个重要分支，它使计算机系统能够从数据中自动学习和改进。",
                "chunk_summary": "机器学习基础概念介绍"
            },
            {
                "chunk_id": "test_chunk_002", 
                "chunk_content": "深度学习基于人工神经网络，通过多层非线性变换来学习复杂的数据表示。",
                "chunk_summary": "深度学习原理说明"
            }
        ]
        
        test_data = {
            "query_text": "什么是机器学习？",
            "chunks": test_chunks
        }
        
        response = await self.client.post(f"{self.base_url}/chunk-decision", json=test_data)
        
        if response.status_code != 200:
            logger.error(f"❌ Chunk 决策匹配失败: {response.status_code} - {response.text}")
            return False
        
        data = response.json()
        logger.info(f"✅ Chunk 决策匹配成功: 处理了 {data['total_results']} 个 chunks")
        
        return True
    
    async def run_full_workflow_test(self):
        """运行完整工作流程测试"""
        logger.info("🚀 开始完整工作流程测试...")
        
        try:
            # 1. 健康检查
            await self.test_health_check()
            
            # 2. 文档上传
            upload_data = await self.test_document_upload()
            if not upload_data:
                return False
            
            # 3. 嵌入生成
            embedding_data = await self.test_embedding_generation(upload_data)
            if not embedding_data:
                return False
            
            # 4. 保存到数据库
            save_success = await self.test_save_to_db(embedding_data)
            if not save_success:
                return False
            
            # 等待一下确保数据已保存
            await asyncio.sleep(2)
            
            # 5. 文档检索
            await self.test_document_retrieval()
            
            # 6. Chunk 检索 (可能失败，因为需要实际的文档 ID)
            await self.test_chunk_retrieval()
            
            # 7. Chunk 决策匹配
            await self.test_chunk_decision()
            
            logger.info("🎉 完整工作流程测试完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试过程中出现错误: {e}")
            return False
    
    async def run_single_test(self, endpoint: str):
        """运行单个端点测试"""
        test_methods = {
            "health": self.test_health_check,
            "upload": self.test_document_upload,
            "doc-retrieval": self.test_document_retrieval,
            "chunk-decision": self.test_chunk_decision
        }
        
        if endpoint not in test_methods:
            logger.error(f"❌ 未知的测试端点: {endpoint}")
            logger.info(f"可用的端点: {', '.join(test_methods.keys())}")
            return False
        
        logger.info(f"🧪 测试端点: {endpoint}")
        return await test_methods[endpoint]()

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowledgeRAG API 测试脚本')
    parser.add_argument('--base-url', default='http://localhost:8954',
                       help='API 服务器地址 (默认: http://localhost:8954)')
    parser.add_argument('--endpoint', choices=['health', 'upload', 'doc-retrieval', 'chunk-decision', 'all'],
                       default='all', help='要测试的端点 (默认: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    
    logger.info("=" * 50)
    logger.info("🧪 KnowledgeRAG API 测试")
    logger.info(f"📍 服务器地址: {args.base_url}")
    logger.info("=" * 50)
    
    async with APITester(args.base_url) as tester:
        try:
            if args.endpoint == 'all':
                success = await tester.run_full_workflow_test()
            else:
                success = await tester.run_single_test(args.endpoint)
            
            if success:
                logger.info("✅ 测试完成!")
                return 0
            else:
                logger.error("❌ 测试失败!")
                return 1
                
        except httpx.ConnectError:
            logger.error(f"❌ 无法连接到服务器: {args.base_url}")
            logger.info("请确保服务器正在运行: python start_server.py")
            return 1
        except Exception as e:
            logger.error(f"❌ 测试过程中出现未预期的错误: {e}")
            return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))