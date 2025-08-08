#!/usr/bin/env python3
"""
简化版检索生成脚本

基于 simple_deep_research.py 的检索和生成部分，提供简单的检索+生成功能。
输入查询，检索相关内容，然后使用LLM生成最终结果。

Author: yujing.wang
Date: 2025.08.05
"""

import json
import os
import logging
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    """生成配置类。
    
    Attributes:
        model_name: 使用的AI模型名称
        temperature: AI模型温度参数
        max_tokens: 最大token数
        rag_api_url: 检索API的URL
    """
    model_name: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4000
    rag_api_url: str = "http://localhost:8001"


class SimpleRetriGenerate:
    """简化的检索生成类。
    
    提供基本的检索和生成功能，不包含深度研究逻辑。
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, config: Optional[GenerateConfig] = None):
        """初始化检索生成模块。
        
        Args:
            openai_api_key: OpenAI API密钥，如果为None则从环境变量获取
            config: 生成配置，如果为None则使用默认配置
        """
        self.config = config or GenerateConfig()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("需要设置 OPENAI_API_KEY 环境变量")
        
        self.client = openai.Client(api_key=self.openai_api_key)
    
    def check_api_health(self) -> bool:
        """检查API健康状态。
        
        Returns:
            API是否健康可用
        """
        try:
            response = requests.get(f"{self.config.rag_api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"API健康检查通过: {health_data.get('documents_count', 0)} 个文档")
                return True
            else:
                logger.error(f"API健康检查失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"API健康检查错误: {e}")
            return False
    
    def search_local_rag(self, query: str) -> Dict[str, Any]:
        """搜索本地 RAG 系统。
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果字典，包含success状态和data/error信息
        """
        try:
            logger.info(f"搜索: {query}")
            
            response = requests.post(
                f"{self.config.rag_api_url}/search/simple",
                json={
                    "query": query,
                    "search_type": "hierarchical",
                    "model": self.config.model_name
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}"}
            
            result = response.json()
            
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "搜索失败")}
            
            logger.info("找到相关信息")
            return {"success": True, "data": result["data"]}
            
        except Exception as e:
            logger.error(f"搜索错误: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_response(self, query: str, search_results: str) -> str:
        """使用LLM生成最终响应。
        
        Args:
            query: 原始查询
            search_results: 检索到的内容
            
        Returns:
            生成的响应文本
        """
        try:
            logger.info("生成响应...")
            
            prompt = f"""You are an expert research assistant. Based on the following query and retrieved information, provide a comprehensive and well-structured response.

Query: {query}

Retrieved Information:
{search_results}

Please provide a detailed response that:
1. Directly addresses the query
2. Synthesizes the retrieved information effectively
3. Provides clear, well-organized information
4. Uses the retrieved content as evidence to support your response
5. Maintains objectivity and accuracy

If the retrieved information is insufficient to fully answer the query, acknowledge this and provide the best possible response based on available information.

Please structure your response clearly with appropriate headings and sections where helpful.
"""
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            if not content:
                return "生成响应时出错：返回内容为空"
            
            return content
            
        except openai.RateLimitError as e:
            logger.error(f"生成响应 - OpenAI API速率限制: {e}")
            return f"生成响应时出错：API速率限制 - {e}"
        except openai.AuthenticationError as e:
            logger.error(f"生成响应 - OpenAI API认证失败: {e}")
            return f"生成响应时出错：API认证失败 - {e}"
        except openai.APIError as e:
            logger.error(f"生成响应 - OpenAI API错误: {e}")
            return f"生成响应时出错：API错误 - {e}"
        except Exception as e:
            logger.error(f"生成响应错误: {e}")
            return f"生成响应时出错: {e}"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """处理查询的完整流程。
        
        Args:
            query: 用户查询
            
        Returns:
            处理结果字典，包含success状态、search_results、generated_response等
        """
        logger.info(f"开始处理查询: {query}")
        
        # 检查API健康状态
        if not self.check_api_health():
            return {
                "success": False,
                "error": "检索API不可用，请确保API服务器已启动",
                "query": query
            }
        
        # 检索阶段
        search_result = self.search_local_rag(query)
        
        if not search_result["success"]:
            return {
                "success": False,
                "error": f"检索失败: {search_result['error']}",
                "query": query
            }
        
        # 格式化检索结果
        search_data = search_result["data"]
        search_results_text = "Here are related documents' summaries:\n\n" + \
                             search_data.get("summary_results", "") + "\n" + \
                             "Here are related documents' chunks:\n\n" + \
                             search_data.get("chunk_results", "")
        
        logger.info(f"检索到相关内容，长度: {len(search_results_text)} 字符")
        
        # 生成阶段
        generated_response = self.generate_response(query, search_results_text)
        
        return {
            "success": True,
            "query": query,
            "search_results": search_results_text,
            "generated_response": generated_response,
            "search_data": search_data  # 包含原始检索数据
        }


def main():
    """主函数，演示脚本使用。
    """
    # 示例查询
    query = "Please systematically summarize the latest research progress in the field of multimodal learning (Multimodal Learning / Multimodal AI)"
    
    # 创建配置
    config = GenerateConfig(
        model_name="gpt-4o",
        temperature=0.1,
        max_tokens=4000,
        rag_api_url="http://localhost:8001"
    )
    
    try:
        # 创建检索生成实例
        retri_generate = SimpleRetriGenerate(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            config=config
        )
        
        # 处理查询
        result = retri_generate.process_query(query)
        
        if result["success"]:
            print("\n" + "=" * 60)
            print("📋 生成的响应")
            print("=" * 60)
            print(result["generated_response"])
            
            print(f"\n📊 处理统计:")
            print(f"   查询: {result['query']}")
            print(f"   检索内容长度: {len(result['search_results'])} 字符")
            print(f"   生成响应长度: {len(result['generated_response'])} 字符")
        else:
            print(f"❌ 处理失败: {result.get('error', '未知错误')}")
        
        return result
        
    except Exception as e:
        print(f"❌ 程序错误: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


def run_simple_retri_generate(query: str, 
                             rag_url: str = "http://localhost:8001",
                             model_name: str = "gpt-4o",
                             temperature: float = 0.1,
                             max_tokens: int = 4000) -> Dict[str, Any]:
    """运行简化检索生成的函数。
    
    Args:
        query: 要处理的查询
        rag_url: 检索API URL (默认: http://localhost:8001)
        model_name: OpenAI模型名称 (默认: "gpt-4o")
        temperature: AI模型温度参数 (默认: 0.1)
        max_tokens: 最大token数 (默认: 4000)
        
    Returns:
        处理结果字典，包含success状态、search_results、generated_response等
    """
    try:
        # 创建配置
        config = GenerateConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            rag_api_url=rag_url
        )
        
        # 创建检索生成实例
        retri_generate = SimpleRetriGenerate(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            config=config
        )
        
        # 处理查询
        result = retri_generate.process_query(query)
        
        return result
        
    except Exception as e:
        logger.error(f"检索生成执行错误: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


# 使用示例
def example_usage():
    """使用示例函数。
    
    展示如何使用不同的配置进行检索生成。
    """
    print("🔍 Simple Retrieval & Generation 使用示例")
    print("=" * 60)
    
    # 示例1: 使用默认配置
    print("\n📝 示例1: 使用默认配置")
    result1 = run_simple_retri_generate(
        query="人工智能在医疗领域的应用"
    )
    print(f"结果: {'成功' if result1['success'] else '失败'}")
    
    # 示例2: 自定义配置
    print("\n📝 示例2: 自定义配置")
    result2 = run_simple_retri_generate(
        query="区块链技术发展趋势",
        model_name="gpt-4o",
        temperature=0.2,
        max_tokens=3000
    )
    print(f"结果: {'成功' if result2['success'] else '失败'}")
    
    # 示例3: 直接使用类
    print("\n📝 示例3: 直接使用类")
    config = GenerateConfig(
        model_name="gpt-4o",
        temperature=0.1,
        max_tokens=2000
    )
    retri_generate = SimpleRetriGenerate(config=config)
    result3 = retri_generate.process_query("机器学习算法比较")
    print(f"结果: {'成功' if result3['success'] else '失败'}")


if __name__ == "__main__":
    # 如果需要运行示例，取消下面的注释
    # example_usage()
    
    # 直接执行并打印结果
    result = main()
    
    if not result.get("success"):
        print(f"❌ 处理失败: {result.get('error', '未知错误')}")





