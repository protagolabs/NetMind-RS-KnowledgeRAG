#!/usr/bin/env python3
"""
KnowledgeRAG API 测试脚本
测试以下三个 API 端点：
1. 按数据集类型检索文档 (/doc-retrieval-by-dataset)
2. Chunk 检索 (/chunk-retrieval)
3. Chunk 决策匹配 (/chunk-decision)

作者: yujing.wang
日期: 2025-08-06
"""

import asyncio
import httpx
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TestResult:
    """测试结果数据类"""
    endpoint: str
    success: bool
    response_time: float
    status_code: int
    response_data: Dict[str, Any]
    error_message: str = ""


class KnowledgeRAGAPITester:
    """KnowledgeRAG API 测试器"""
    
    def __init__(self, base_url: str = "http://71.178.110.3:8955"):
        self.base_url = base_url
        self.client = None
        self.test_results: List[TestResult] = []
        # 存储中间结果
        self.doc_ids = []
        self.chunks = []
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.client:
            await self.client.aclose()
    
    async def test_health_check(self) -> bool:
        """测试健康检查"""
        try:
            response = await self.client.get(f"{self.base_url}/")
            if response.status_code == 200:
                print("✅ 健康检查通过")
                return True
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 健康检查异常: {e}")
            return False
    
    async def test_dataset_document_retrieval(self) -> TestResult:
        """测试按数据集类型检索文档 - 第一步"""
        endpoint = "/doc-retrieval-by-dataset"
        start_time = time.time()
        
        # 测试用例
        test_cases = [
            {
                "name": "机器学习数据集检索",
                "data": {
                    "data_set_type": "llm_papers",
                    "query": "What is a multimodal large language model?"
                }
            },
            # {
            #     "name": "自然语言处理数据集检索", 
            #     "data": {
            #         "data_set_type": "paper_set_2",
            #         "query": "Transformer模型在NLP中的应用"
            #     }
            # },
            # {
            #     "name": "通用数据集检索",
            #     "data": {
            #         "data_set_type": "general",
            #         "query": "人工智能技术发展"
            #     }
            # }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 测试用例 {i}: {test_case['name']}")
            print(f"   数据集类型: {test_case['data']['data_set_type']}")
            print(f"   查询内容: {test_case['data']['query']}")
            
            try:
                response = await self.client.post(
                    f"{self.base_url}{endpoint}",
                    json=test_case['data']
                )
                
                response_time = time.time() - start_time
                response_data = response.json()
                
                result = TestResult(
                    endpoint=endpoint,
                    success=response.status_code == 200 and response_data.get('success', False),
                    response_time=response_time,
                    status_code=response.status_code,
                    response_data=response_data
                )
                
                if result.success:
                    print(f"   ✅ 成功 - 响应时间: {response_time:.2f}s")
                    print(f"   找到 {response_data.get('total_results', 0)} 个结果")
                    
                    # 提取 doc_ids 用于下一步
                    results = response_data.get('results', [])
                    doc_ids = [doc.get('file_id') for doc in results if doc.get('file_id')]
                    self.doc_ids.extend(doc_ids)
                    
                    print(f"   📋 提取到 {len(doc_ids)} 个文档ID: {doc_ids}")
                    
                    # 显示前3个结果
                    for j, doc in enumerate(results[:3], 1):
                        print(f"   📄 结果 {j}: {doc.get('file_name', 'Unknown')}")
                        print(f"      相关性分数: {doc.get('relevance_score', 0):.3f}")
                        print(f"      摘要: {doc.get('summary', '')[:100]}...")
                else:
                    print(f"   ❌ 失败 - 状态码: {response.status_code}")
                    print(f"   错误信息: {response_data.get('message', 'Unknown error')}")
                
                self.test_results.append(result)
                
            except Exception as e:
                response_time = time.time() - start_time
                result = TestResult(
                    endpoint=endpoint,
                    success=False,
                    response_time=response_time,
                    status_code=0,
                    response_data={},
                    error_message=str(e)
                )
                print(f"   ❌ 异常: {e}")
                self.test_results.append(result)
        
        return result
    
    async def test_chunk_retrieval(self) -> TestResult:
        """测试 Chunk 检索 - 第二步"""
        endpoint = "/chunk-retrieval"
        start_time = time.time()
        
        if not self.doc_ids:
            print("❌ 没有可用的文档ID，跳过Chunk检索测试")
            return TestResult(
                endpoint=endpoint,
                success=False,
                response_time=0,
                status_code=0,
                response_data={},
                error_message="No doc_ids available"
            )
        print(f"self.doc_ids: {self.doc_ids}")
        # 使用第一步获取的 doc_ids
        test_cases = [
            {
                "name": "神经网络优化方法检索",
                "data": {
                    "query": "What is a multimodal large language model?",
                    "doc_ids": self.doc_ids, 
                    "each_doc_chunk_number": 10
                }
            },
            # {
            #     "name": "深度学习模型架构检索",
            #     "data": {
            #         "query": "卷积神经网络架构设计",
            #         "doc_ids": self.doc_ids[:2],  # 使用前2个文档ID
            #         "each_doc_chunk_number": 10
            #     }
            # },
            # {
            #     "name": "机器学习算法检索",
            #     "data": {
            #         "query": "支持向量机算法原理",
            #         "doc_ids": self.doc_ids[:1],  # 使用前1个文档ID
            #         "each_doc_chunk_number": 3
            #     }
            # }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 测试用例 {i}: {test_case['name']}")
            print(f"   查询内容: {test_case['data']['query']}")
            print(f"   文档ID: {test_case['data']['doc_ids']}")
            print(f"   每个文档chunk数量: {test_case['data']['each_doc_chunk_number']}")
            
            try:
                response = await self.client.post(
                    f"{self.base_url}{endpoint}",
                    json=test_case['data']
                )
                
                response_time = time.time() - start_time
                response_data = response.json()
                
                result = TestResult(
                    endpoint=endpoint,
                    success=response.status_code == 200 and response_data.get('success', False),
                    response_time=response_time,
                    status_code=response.status_code,
                    response_data=response_data
                )
                
                if result.success:
                    print(f"   ✅ 成功 - 响应时间: {response_time:.2f}s")
                    print(f"   找到 {response_data.get('total_results', 0)} 个chunks")
                    
                    # 提取 chunks 用于下一步
                    results = response_data.get('results', [])
                    
                    # 打印一下results的json格式
                    # formatted_results = json.dumps(results, ensure_ascii=False, indent=2)
                    # print(f"   📋 提取到的结果:\n{formatted_results}")
                    
                    
                    chunks = []
                    for chunk in results:
                        chunk_data = {
                            "chunk_id": chunk.get('chunk_id'),
                            "summary": chunk.get('summary', ''),
                            "insights": chunk.get('insights', ''),
                            "key_words": chunk.get('key_words', []),
                            "chunk_markdown_content": chunk.get('chunk_markdown_content', ''),
                        }
                        chunks.append(chunk_data)
                    
                    self.chunks.extend(chunks)
                    print(f"   📋 提取到 {len(chunks)} 个chunks")
                    
                    # 显示前3个结果
                    for j, chunk in enumerate(results[:3], 1):
                        print(f"   📄 Chunk {j}: {chunk.get('chunk_id', 'Unknown')}")
                        print(f"      相似度分数: {chunk.get('similarity_score', 0):.3f}")
                        print(f"      来源文档: {chunk.get('source_document', 'Unknown')}")
                        print(f"      内容摘要: {chunk.get('chunk_summary', '')[:80]}...")
                else:
                    print(f"   ❌ 失败 - 状态码: {response.status_code}")
                    print(f"   错误信息: {response_data.get('message', 'Unknown error')}")
                
                self.test_results.append(result)
                
            except Exception as e:
                response_time = time.time() - start_time
                result = TestResult(
                    endpoint=endpoint,
                    success=False,
                    response_time=response_time,
                    status_code=0,
                    response_data={},
                    error_message=str(e)
                )
                print(f"   ❌ 异常: {e}")
                self.test_results.append(result)
        
        return result
    
    async def test_chunk_decision(self) -> TestResult:
        """测试 Chunk 决策匹配 - 第三步"""
        endpoint = "/chunk-decision"
        start_time = time.time()
        
        if not self.chunks:
            print("❌ 没有可用的chunks，跳过Chunk决策测试")
            return TestResult(
                endpoint=endpoint,
                success=False,
                response_time=0,
                status_code=0,
                response_data={},
                error_message="No chunks available"
            )
        # # 打印一下self.chunks
        # print(f"self.chunks: {self.chunks}")
        
        # 使用第二步获取的 chunks
        test_cases = [
            {
                "name": "机器学习模型评估决策",
                "data": {
                    "query_text": "What is a multimodal large language model?",
                    "chunks": self.chunks  # 使用前3个chunks
                }
            },
            # {
            #     "name": "神经网络优化决策",
            #     "data": {
            #         "query_text": "神经网络优化算法",
            #         "chunks": self.chunks[3:6] if len(self.chunks) >= 6 else self.chunks  # 使用接下来的chunks
            #     }
            # }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 测试用例 {i}: {test_case['name']}")
            print(f"   查询文本: {test_case['data']['query_text']}")
            print(f"   Chunk数量: {len(test_case['data']['chunks'])}")
            
            try:
                response = await self.client.post(
                    f"{self.base_url}{endpoint}",
                    json=test_case['data']
                )
                
                response_time = time.time() - start_time
                response_data = response.json()
                
                result = TestResult(
                    endpoint=endpoint,
                    success=response.status_code == 200 and response_data.get('success', False),
                    response_time=response_time,
                    status_code=response.status_code,
                    response_data=response_data
                )
                
                if result.success:
                    print(f"   ✅ 成功 - 响应时间: {response_time:.2f}s")
                    print(f"   决策结果数量: {response_data.get('total_results', 0)}")
                    
                    # 显示所有决策结果
                    results = response_data.get('results', [])
                    for j, decision in enumerate(results, 1):
                        print(f"   📄 决策 {j}: {decision.get('chunk_id', 'Unknown')}")
                        print(f"      相关性分数: {decision.get('relevance_score', 0):.3f}")
                        print(f"      是否相关: {'✅' if decision.get('is_relevant', False) else '❌'}")
                        print(f"      决策原因: {decision.get('decision_reason', '')}")
                        print(f"      内容摘要: {decision.get('chunk_content', '')[:80]}...")
                else:
                    print(f"   ❌ 失败 - 状态码: {response.status_code}")
                    print(f"   错误信息: {response_data.get('message', 'Unknown error')}")
                
                self.test_results.append(result)
                
            except Exception as e:
                response_time = time.time() - start_time
                result = TestResult(
                    endpoint=endpoint,
                    success=False,
                    response_time=response_time,
                    status_code=0,
                    response_data={},
                    error_message=str(e)
                )
                print(f"   ❌ 异常: {e}")
                self.test_results.append(result)
        
        return result
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("📊 测试总结")
        print("="*60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.success)
        
        print(f"总测试数: {total_tests}")
        print(f"成功测试: {successful_tests}")
        print(f"失败测试: {total_tests - successful_tests}")
        print(f"成功率: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "成功率: 0%")
        
        if self.test_results:
            avg_response_time = sum(result.response_time for result in self.test_results) / len(self.test_results)
            print(f"平均响应时间: {avg_response_time:.2f}s")
        
        print(f"\n📋 数据流转:")
        print(f"   第一步获取文档ID数量: {len(self.doc_ids)}")
        print(f"   第二步获取Chunk数量: {len(self.chunks)}")
        
        print("\n详细结果:")
        for i, result in enumerate(self.test_results, 1):
            status = "✅ 成功" if result.success else "❌ 失败"
            print(f"{i}. {result.endpoint}: {status} ({result.response_time:.2f}s)")
            if not result.success and result.error_message:
                print(f"   错误: {result.error_message}")


async def main():
    """主函数"""
    print("🚀 KnowledgeRAG API 串联测试开始")
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    async with KnowledgeRAGAPITester() as tester:
        # 健康检查
        if not await tester.test_health_check():
            print("❌ 服务不可用，停止测试")
            return
        
        print("\n" + "="*60)
        print("🧪 开始串联 API 端点测试")
        print("="*60)
        
        # 测试1: 按数据集类型检索文档
        print("\n📋 第一步: 按数据集类型检索文档")
        print("-" * 40)
        await tester.test_dataset_document_retrieval()
        
        # 测试2: Chunk 检索
        print("\n📋 第二步: Chunk 检索")
        print("-" * 40)
        await tester.test_chunk_retrieval()
        
        # 测试3: Chunk 决策匹配
        print("\n📋 第三步: Chunk 决策匹配")
        print("-" * 40)
        await tester.test_chunk_decision()
        
        # 打印总结
        tester.print_summary()
    
    print("\n🎉 串联测试完成!")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main()) 