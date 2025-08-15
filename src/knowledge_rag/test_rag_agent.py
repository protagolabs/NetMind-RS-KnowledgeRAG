#!/usr/bin/env python3
"""
RAG Agent 测试脚本

功能：
1. 读取处理后的QA数据
2. 对每个query测试RAG agent
3. 记录详细的性能指标：
   - 每个query的处理时间
   - 每个query的费用信息
   - 使用的chunk IDs
   - 生成的答案
4. 生成测试报告
"""

import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from knowledge_rag.agentic_generation.rag_agent import RAGAgent


class RAGAgentTester:
    """RAG Agent 测试器"""
    
    def __init__(self, processed_data_path: str):
        """
        初始化测试器
        
        Args:
            processed_data_path: 处理后的QA数据文件路径
        """
        self.processed_data_path = processed_data_path
        self.rag_agent = RAGAgent()
        self.test_results = []
        
    def load_test_data(self) -> Dict[str, Any]:
        """加载测试数据"""
        print(f"正在加载测试数据: {self.processed_data_path}")
        with open(self.processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    async def test_single_query(self, query: str, expected_answer: str = None, 
                               chunk_info: List[Dict] = None, question_type: str = None,
                               description: str = None) -> Dict[str, Any]:
        """
        测试单个查询
        
        Args:
            query: 查询文本
            expected_answer: 期望的答案（可选）
            chunk_info: chunk信息（可选）
            question_type: 问题类型
            description: 问题描述
            
        Returns:
            测试结果字典
        """
        print(f"\n正在测试查询: {query[:100]}...")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 调用RAG agent
            rag_result = await self.rag_agent.rag_agent(query)
            
            # 记录结束时间
            end_time = time.time()
            total_time = end_time - start_time
            
            # 从RAG agent结果中提取信息
            generated_answer = rag_result.get("answer", "")
            chunk_ids = rag_result.get("chunk_ids", [])
            cost_breakdown = rag_result.get("cost_breakdown", {})
            performance_metrics = rag_result.get("performance_metrics", {})
            metadata = rag_result.get("metadata", {})
            
            # 构建测试结果
            test_result = {
                "query": query,
                "generated_answer": generated_answer,
                "total_time_seconds": total_time,
                "timestamp": datetime.now().isoformat(),
                "expected_answer": expected_answer,
                "chunk_info": chunk_info,
                "question_type": question_type,
                "description": description,
                "status": "success",
                "chunks_used": chunk_ids,
                "cost_breakdown": cost_breakdown,
                "performance_metrics": performance_metrics,
                "metadata": metadata
            }
            
            print(f"✅ 测试成功 - 耗时: {total_time:.2f}秒")
            print(f"生成答案: {generated_answer[:200]}...")
            print(f"使用chunks: {len(chunk_ids)} 个")
            print(f"费用: ${cost_breakdown.get('total_cost_usd', 0):.4f}")
            
        except Exception as e:
            # 记录错误
            end_time = time.time()
            total_time = end_time - start_time
            
            test_result = {
                "query": query,
                "generated_answer": None,
                "total_time_seconds": total_time,
                "timestamp": datetime.now().isoformat(),
                "expected_answer": expected_answer,
                "chunk_info": chunk_info,
                "question_type": question_type,
                "description": description,
                "status": "error",
                "error_message": str(e),
                "chunks_used": [],
                "cost_breakdown": {
                    "total_cost_usd": 0,
                    "intent_recognition_cost": 0,
                    "query_rewrite_cost": 0,
                    "generation_cost": 0
                },
                "performance_metrics": {},
                "metadata": {}
            }
            
            print(f"❌ 测试失败 - 耗时: {total_time:.2f}秒")
            print(f"错误信息: {e}")
        
        return test_result
    
    def extract_queries_from_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从数据中提取所有查询
        
        Args:
            data: 处理后的QA数据
            
        Returns:
            查询列表，每个元素包含query、expected_answer和chunk_info
        """
        queries = []
        
        # 遍历每个大类（single_doc_multi_chunk, multi_doc_multi_chunk）
        for category_name, category_data in data.items():
            print(f"\n处理类别: {category_name}")
            
            # 遍历每个问题类型
            for question_type, questions in category_data.items():
                print(f"  问题类型: {question_type} ({len(questions)} 个问题)")
                
                for question in questions:
                    query_info = {
                        "query": question.get("question", ""),
                        "expected_answer": question.get("answer", ""),
                        "chunk_info": question.get("chunk", []),
                        "question_type": question.get("question_type", ""),
                        "category_name": category_name,  # 大类名称
                        "description": question.get("des", "")
                    }
                    queries.append(query_info)
        
        print(f"\n总共提取了 {len(queries)} 个查询")
        return queries
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """
        运行所有测试
        
        Returns:
            所有测试结果列表
        """
        # 加载测试数据
        data = self.load_test_data()
        
        # 提取所有查询
        queries = self.extract_queries_from_data(data)
        
        # 运行测试
        print(f"\n开始运行 {len(queries)} 个测试...")
        # 先测试两个问题
        for i, query_info in enumerate(queries[:2], 1):
            print(f"\n--- 测试 {i}/{len(queries)} ---")
            print(f"类别: {query_info['category_name']}")
            print(f"问题类型: {query_info['question_type']}")
            
            result = await self.test_single_query(
                query=query_info["query"],
                expected_answer=query_info["expected_answer"],
                chunk_info=query_info["chunk_info"],
                question_type=query_info["question_type"],
                description=query_info["description"]
            )
            
            # 添加额外信息
            result.update({
                "category_name": query_info["category_name"],
                "test_index": i
            })
            
            self.test_results.append(result)
            
            # 添加延迟避免API限制
            await asyncio.sleep(1)
        
        return self.test_results
    
    def generate_report(self, output_path: str = None):
        """
        生成测试报告
        
        Args:
            output_path: 输出文件路径
        """
        if not self.test_results:
            print("没有测试结果可生成报告")
            return
        
        # 计算统计信息
        total_queries = len(self.test_results)
        successful_queries = len([r for r in self.test_results if r["status"] == "success"])
        failed_queries = total_queries - successful_queries
        
        total_time = sum(r["total_time_seconds"] for r in self.test_results)
        avg_time = total_time / total_queries if total_queries > 0 else 0
        
        # 计算总费用
        total_cost = sum(r.get("cost_breakdown", {}).get("total_cost_usd", 0) for r in self.test_results)
        avg_cost = total_cost / total_queries if total_queries > 0 else 0
        
        # 按类别统计
        category_stats = {}
        for result in self.test_results:
            category = result.get("category_name", "Unknown")
            if category not in category_stats:
                category_stats[category] = {"total": 0, "success": 0, "failed": 0, "avg_time": 0, "total_cost": 0}
            
            category_stats[category]["total"] += 1
            if result["status"] == "success":
                category_stats[category]["success"] += 1
            else:
                category_stats[category]["failed"] += 1
            
            # 累加费用
            category_stats[category]["total_cost"] += result.get("cost_breakdown", {}).get("total_cost_usd", 0)
        
        # 计算每个类别的平均时间和平均费用
        for category in category_stats:
            category_results = [r for r in self.test_results if r.get("category_name") == category]
            if category_results:
                avg_time_category = sum(r["total_time_seconds"] for r in category_results) / len(category_results)
                avg_cost_category = category_stats[category]["total_cost"] / len(category_results)
                category_stats[category]["avg_time"] = avg_time_category
                category_stats[category]["avg_cost"] = avg_cost_category
        
        # 按问题类型统计
        type_stats = {}
        for result in self.test_results:
            qtype = result.get("question_type", "Unknown")
            if qtype not in type_stats:
                type_stats[qtype] = {"total": 0, "success": 0, "failed": 0, "avg_time": 0, "total_cost": 0}
            
            type_stats[qtype]["total"] += 1
            if result["status"] == "success":
                type_stats[qtype]["success"] += 1
            else:
                type_stats[qtype]["failed"] += 1
            
            # 累加费用
            type_stats[qtype]["total_cost"] += result.get("cost_breakdown", {}).get("total_cost_usd", 0)
        
        # 计算每个问题类型的平均时间和平均费用
        for qtype in type_stats:
            type_results = [r for r in self.test_results if r.get("question_type") == qtype]
            if type_results:
                avg_time_type = sum(r["total_time_seconds"] for r in type_results) / len(type_results)
                avg_cost_type = type_stats[qtype]["total_cost"] / len(type_results)
                type_stats[qtype]["avg_time"] = avg_time_type
                type_stats[qtype]["avg_cost"] = avg_cost_type
        
        # 生成报告
        report = {
            "summary": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
                "total_time_seconds": total_time,
                "average_time_seconds": avg_time,
                "total_cost_usd": total_cost,
                "average_cost_usd": avg_cost,
                "test_timestamp": datetime.now().isoformat()
            },
            "category_statistics": category_stats,
            "question_type_statistics": type_stats,
            "detailed_results": self.test_results
        }
        
        # 保存报告
        if output_path is None:
            output_path = "rag_agent_test_report.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        print(f"\n{'='*50}")
        print("测试报告摘要")
        print(f"{'='*50}")
        print(f"总查询数: {total_queries}")
        print(f"成功查询: {successful_queries}")
        print(f"失败查询: {failed_queries}")
        print(f"成功率: {successful_queries/total_queries*100:.1f}%")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均耗时: {avg_time:.2f}秒")
        print(f"总费用: ${total_cost:.4f}")
        print(f"平均费用: ${avg_cost:.4f}")
        
        print(f"\n按类别统计:")
        for category, stats in category_stats.items():
            success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {category}: {stats['success']}/{stats['total']} ({success_rate:.1f}%) - 平均耗时: {stats['avg_time']:.2f}秒, 平均费用: ${stats['avg_cost']:.4f}")
        
        print(f"\n按问题类型统计:")
        for qtype, stats in type_stats.items():
            success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {qtype}: {stats['success']}/{stats['total']} ({success_rate:.1f}%) - 平均耗时: {stats['avg_time']:.2f}秒, 平均费用: ${stats['avg_cost']:.4f}")
        
        print(f"\n详细报告已保存到: {output_path}")


async def main():
    """主函数"""
    # 文件路径 - 使用正确的路径
    processed_data_path = "processed_qa_data.json"  # 文件在同一目录下
    report_path = "rag_agent_test_report.json"
    
    # 检查输入文件是否存在
    if not Path(processed_data_path).exists():
        print(f"错误: 输入文件不存在: {processed_data_path}")
        print("请确保已经运行了 process_qa_data.py 脚本")
        print("或者检查文件路径是否正确")
        print(f"当前工作目录: {Path.cwd()}")
        print(f"尝试查找文件: {Path(processed_data_path).absolute()}")
        return
    
    # 创建测试器
    tester = RAGAgentTester(processed_data_path)
    
    try:
        # 运行所有测试
        results = await tester.run_all_tests()
        
        # 生成报告
        tester.generate_report(report_path)
        
        print(f"\n🎉 测试完成! 共测试了 {len(results)} 个查询")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 