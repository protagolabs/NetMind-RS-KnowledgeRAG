""" 
@file_name: openai_client.py
@author: Yujing Wang, Bin Liang
@date: 2025-08-11
@description: 
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
import traceback

from knowledge_rag.config import OPENAI_API_KEY


class OpenAICostCalculator:
    """OpenAI费用计算器。
    
    根据GPT-4.1的收费标准计算费用：
    - 输入：每1M tokens $2.00
    - 输出：每1M tokens $8.00
    """
    
    def __init__(self):
        """初始化费用计算器。"""
        self.cost_table = {
            "gpt-4.1": {          # $2 / 1M in, $8 / 1M out
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.000008
            },
            "gpt-4o": {           # $5 / 1M in, $20 / 1M out
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.000020
            },
            "gpt-4o-mini": {      # $0.60 / 1M in, $2.40 / 1M out
                "input_cost_per_token": 0.0000006,
                "output_cost_per_token": 0.0000024
            },
            "gpt-5": {
                "input_cost_per_token": 0.00000125,
                "output_cost_per_token": 0.00001
            }
        }
        
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> Dict[str, float]:
        """计算费用。
        
        Args:
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            
        Returns:
            包含各项费用的字典
        """
        
        for key, value in self.cost_table.items():
            if model.startswith(key):
                model = key
                break
        
        input_cost = input_tokens * self.cost_table[model]["input_cost_per_token"]
        output_cost = output_tokens * self.cost_table[model]["output_cost_per_token"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6)
        }
    
    
class OpenAIClient:
    """OpenAI客户端包装类。
    
    提供统一的OpenAI API调用接口，支持同步和异步调用。
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """初始化OpenAI包装器。
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
        """
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("需要设置 OPENAI_API_KEY 环境变量")

        self.cost_calculator = OpenAICostCalculator()
    
    async def close(self):
        """关闭客户端连接"""
        if hasattr(self, 'client') and self.client:
            await self.client.close()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            if hasattr(self, 'client') and self.client:
                # 不能在__del__中调用异步方法，只能尝试同步关闭
                try:
                    # 尝试访问底层的httpx客户端并关闭
                    if hasattr(self.client, '_client') and hasattr(self.client._client, 'close'):
                        self.client._client.close()
                except:
                    pass
        except:
            pass
    
    def _handle_openai_error(self, error: Exception, operation: str) -> str:
        """处理OpenAI API错误。
        
        Args:
            error: 错误对象
            operation: 操作名称
            
        Returns:
            错误信息字符串
        """
        if "rate_limit" in str(error).lower():
            return f"{operation} - OpenAI API速率限制，请稍后重试"
        elif "quota" in str(error).lower():
            return f"{operation} - OpenAI API配额已用完"
        elif "authentication" in str(error).lower():
            return f"{operation} - OpenAI API认证失败，请检查API密钥"
        else:
            return f"{operation} - OpenAI API错误: {error}"
    
    async def parse_structured_response(
        self, 
        model: str,
        messages: List,
        response_model: BaseModel,
        temperature: float = 1,
        **kwargs
    ) -> Optional[Tuple[BaseModel, Dict[str, float]]]:
        """异步解析结构化响应。
        
        Args:
            model: 模型名称
            messages: 消息列表
            response_model: 响应模型类
            temperature: 温度参数
            
        Returns:
            解析后的模型对象和费用信息的元组，如果失败则返回None
        """
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            completion = await client.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_model, # type: ignore
                **kwargs
            )
            
            # 计算费用
            input_tokens = completion.usage.prompt_tokens # type: ignore
            output_tokens = completion.usage.completion_tokens # type: ignore
            cost_info = self.calculate_cost(input_tokens, output_tokens, model)
            
            return completion.choices[0].message.parsed, cost_info # type: ignore
        except Exception as e:
            import traceback 
            error_message = traceback.format_exc()
            print(error_message)
            raise ValueError(f"Error in parse_structured_response: {error_message}")
    
    async def generate_text(
        self,
        model: str,
        messages: List,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """异步生成文本。
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            生成的文本和费用信息的元组，如果失败则返回None
        """
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            completion = await client.chat.completions.create(
                model=model,
                messages=messages, # type: ignore
                temperature=temperature,
                **kwargs
            )
            
            completion_text = completion.choices[0].message.content or "" # type: ignore
            input_tokens = completion.usage.prompt_tokens # type: ignore
            output_tokens = completion.usage.completion_tokens # type: ignore
            
            # 计算费用
            cost_info = self.calculate_cost(input_tokens, output_tokens, model)
            
            return completion_text, cost_info
        except Exception as e:
            error_msg = traceback.format_exc()
            raise ValueError(f"Error in generate_text: {error_msg}")
    
    def calculate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int, 
        model: str = "gpt-4.1"
    ) -> Dict[str, float]:
        """计算OpenAI API调用的费用。
        
        Args:
            messages: 消息列表
            completion_text: 完成文本
            
        Returns:
            包含各项费用的字典
        """
        return self.cost_calculator.calculate_cost(input_tokens, output_tokens, model)
    

# --- unit test ---
# async def unit_test_structured_response():
#     """
#     """
#     openai_client = OpenAIClient()
    
#     class Jokes(BaseModel):
#         jokes: List[str]
    
#     result = await openai_client.parse_structured_response(
#         model="gpt-4.1",
#         messages=[{"role": "user", "content": "给我讲几个笑话。"}],
#         response_model=Jokes
#     )
    
#     print(result)
    
# async def unit_test_generate_text():
#     """
#     """
#     openai_client = OpenAIClient()
#     result = await openai_client.generate_text(
#         model="gpt-4.1",
#         messages=[{"role": "user", "content": "给我讲几个笑话。"}],
#         temperature=0.1,
#         max_tokens=1000
#     )
    
#     print(result)
    
    
# if __name__ == "__main__":
#     import asyncio
#     # asyncio.run(unit_test_structured_response())
#     asyncio.run(unit_test_generate_text())
    
    