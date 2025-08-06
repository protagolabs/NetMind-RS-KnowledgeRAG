#!/usr/bin/env python3
"""
文档检索请求示例
================

作者: Bin Liang
日期: 2025-08-01
描述: 演示如何使用 Python 调用 KnowledgeRAG API 的文档检索功能

使用方法:
    python example_doc_retrieval.py
"""

import asyncio
import httpx
import json
from typing import Dict, Any

async def document_retrieval_example():
    """文档检索请求示例"""
    
    # API 服务器地址
    base_url = "http://localhost:8955"  # 使用你修改的端口
    
    # 构建请求数据
    request_data = {
        "query": "Please walk me through the changes in model training methods"
    }
    
    print("🔍 发送文档检索请求...")
    print(f"📍 服务器地址: {base_url}")
    print(f"🎯 查询内容: {request_data['query']}")
    print("-" * 50)
    
    try:
        # 创建异步 HTTP 客户端
        async with httpx.AsyncClient(timeout=30.0) as client:
            
            # 发送 POST 请求到文档检索端点
            response = await client.post(
                f"{base_url}/doc-retrieval",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            # 检查响应状态
            if response.status_code == 200:
                # 解析 JSON 响应
                result = response.json()
                
                print("✅ 请求成功!")
                print(f"📊 状态: {result['success']}")
                print(f"💬 消息: {result['message']}")
                print(f"📈 结果数量: {result['total_results']}")
                print()
                
                # 显示检索结果
                if result['results'] and len(result['results']) > 0:
                    print("📋 检索结果:")
                    print("=" * 60)
                    
                    for i, doc in enumerate(result['results'][:5], 1):  # 显示前5个结果
                        print(f"\n🔸 结果 {i}:")
                        
                        # 根据实际返回的数据结构显示信息
                        if isinstance(doc, dict):
                            # 如果是字典格式，显示可用字段
                            for key, value in doc.items():
                                if key in ['file_name', 'summary', 'relevance_score', 'match_reason']:
                                    print(f"   {key}: {value}")
                                elif key == 'file_id':
                                    print(f"   文档ID: {value}")
                        else:
                            # 如果是其他格式，直接显示
                            print(f"   内容: {doc}")
                        
                        print("-" * 40)
                else:
                    print("⚠️ 没有找到相关文档")
                
                return result
                
            else:
                # 处理错误响应
                print(f"❌ 请求失败!")
                print(f"📊 状态码: {response.status_code}")
                print(f"💬 错误信息: {response.text}")
                
                try:
                    error_data = response.json()
                    if 'message' in error_data:
                        print(f"🔍 详细错误: {error_data['message']}")
                except:
                    pass
                
                return None
                
    except httpx.ConnectError:
        print("❌ 连接失败!")
        print("请确保 API 服务器正在运行:")
        print(f"   python start_server.py --port 8955")
        return None
        
    except httpx.TimeoutException:
        print("❌ 请求超时!")
        print("服务器响应时间过长，请稍后重试")
        return None
        
    except Exception as e:
        print(f"❌ 发生未预期的错误: {e}")
        return None

async def multiple_queries_example():
    """多个查询示例"""
    
    queries = [
        "Please walk me through the changes in model training methods",
        "What are the latest advances in deep learning?",
        "How has neural network architecture evolved?",
        "What are the key improvements in optimization algorithms?"
    ]
    
    print("🎯 批量查询示例")
    print("=" * 50)
    
    base_url = "http://localhost:8955"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        for i, query in enumerate(queries, 1):
            print(f"\n📝 查询 {i}: {query}")
            print("-" * 30)
            
            try:
                response = await client.post(
                    f"{base_url}/doc-retrieval",
                    json={"query": query}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 找到 {result['total_results']} 个相关文档")
                    
                    # 显示第一个结果的摘要
                    if result['results']:
                        first_result = result['results'][0]
                        if isinstance(first_result, dict) and 'summary' in first_result:
                            summary = first_result['summary'][:100] + "..." if len(first_result['summary']) > 100 else first_result['summary']
                            print(f"📄 首个结果摘要: {summary}")
                else:
                    print(f"❌ 查询失败: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 查询出错: {e}")
            
            # 避免请求过快
            await asyncio.sleep(0.5)

def sync_request_example():
    """同步请求示例 (使用 requests 库)"""
    
    try:
        import requests
    except ImportError:
        print("❌ 需要安装 requests 库: pip install requests")
        return
    
    print("🔄 同步请求示例")
    print("=" * 30)
    
    base_url = "http://localhost:8955"
    
    request_data = {
        "query": "Please walk me through the changes in model training methods"
    }
    
    try:
        response = requests.post(
            f"{base_url}/doc-retrieval",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 同步请求成功!")
            print(f"📈 找到 {result['total_results']} 个结果")
        else:
            print(f"❌ 同步请求失败: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败 (同步)")
    except Exception as e:
        print(f"❌ 同步请求出错: {e}")

async def main():
    """主函数"""
    print("🚀 KnowledgeRAG 文档检索示例")
    print("=" * 50)
    
    # 1. 单个查询示例
    print("\n1️⃣ 单个查询示例:")
    await document_retrieval_example()
    
    # 等待一下
    await asyncio.sleep(1)
    
    # 2. 多个查询示例
    print("\n\n2️⃣ 多个查询示例:")
    await multiple_queries_example()
    
    # 3. 同步请求示例
    print("\n\n3️⃣ 同步请求示例:")
    sync_request_example()
    
    print("\n🎉 示例运行完成!")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())