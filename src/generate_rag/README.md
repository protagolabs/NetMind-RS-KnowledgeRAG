# Complex Retrieval Generation System

基于深度研究的复杂检索生成系统，实现四步检索生成流程，提供高质量的RAG（检索增强生成）解决方案。

## 🚀 功能特性

### 四步检索生成流程
1. **意图识别** - 判断查询是信息汇总型(1)还是事实型(0)
2. **查询改写** - 基于文档总结改写查询以增加召回
3. **检索** - 使用KnowledgeRAG API检索多个查询
4. **生成** - 基于检索结果生成最终答案

### 核心优势
- ✅ **智能意图识别** - 自动判断查询类型，优化检索策略
- ✅ **多查询改写** - 生成多个改写版本，提高召回率
- ✅ **结构化输出** - 支持引用格式，便于追踪信息来源
- ✅ **费用计算** - 实时计算OpenAI API调用费用
- ✅ **详细日志** - 完整的处理步骤记录和错误追踪
- ✅ **结果保存** - 自动保存Markdown和JSON格式结果

## 📋 系统要求

### Python版本
- Python 3.8+

### 依赖包
```bash
pip install openai httpx tiktoken pydantic requests asyncio
```

### 环境变量
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 🔧 配置说明

### ComplexRetriConfig 配置类
```python
@dataclass
class ComplexRetriConfig:
    rag_api_url: str = "http://localhost:8001"                    # RAG API地址
    knowledge_rag_api_url: str = "http://71.178.110.3:8955"       # KnowledgeRAG API地址
    model_name: str = "gpt-4.1"                                   # OpenAI模型名称
    temperature: float = 0.1                                       # 模型温度参数
    max_tokens: int = 4000                                         # 最大输出token数
    max_queries: int = 5                                           # 最大改写查询数量
```

## 🎯 使用方法

### 1. 基本使用

```python
import asyncio
from complex_retri_generate import ComplexRetriConfig, ComplexRetriWorkflow

async def main():
    # 创建配置
    config = ComplexRetriConfig(
        model_name="gpt-4o-mini",  # 使用便宜的模型
        max_queries=3,              # 减少查询数量
        max_tokens=2000
    )
    
    # 创建工作流
    workflow = ComplexRetriWorkflow(config)
    
    # 处理查询
    query = "What is a large language model?"
    result = await workflow.process(query)
    
    if result["success"]:
        print(f"✅ 处理成功")
        print(f"📊 意图类型: {'信息汇总型' if result['intent']['intent_type'] == 1 else '事实型'}")
        print(f"🔄 改写查询数: {result['rewritten_queries']}")
        print(f"🔍 检索成功率: {result['retrieval_success_rate']}")
        print(f"⏱️ 总耗时: {result['total_time']:.2f}秒")
        print(f"💰 总费用: ${result['total_cost_usd']:.6f} USD")
        print(f"📄 Markdown文件: {result['markdown_file']}")
        print(f"📊 RAG结果文件: {result['rag_result_file']}")
        
        # 显示最终答案
        print(f"\n📝 最终答案:")
        print(result['final_answer'])
    else:
        print(f"❌ 处理失败: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
```


## 📊 输出结果

### 处理结果结构
```python
{
    "success": True,
    "query": "原始查询",
    "intent": {
        "intent_type": 1,  # 0=事实型, 1=信息汇总型
        "reasoning": "推理过程"
    },
    "rewritten_queries": 3,
    "retrieval_success_rate": "3/3",
    "final_answer": "生成的最终答案",
    "total_time": 45.23,
    "total_cost_usd": 0.123456,
    "step_costs": {
        "意图识别": {"total_cost_usd": 0.001234, "input_tokens": 50, "output_tokens": 20},
        "查询改写": {"total_cost_usd": 0.012345, "input_tokens": 200, "output_tokens": 150},
        "生成": {"total_cost_usd": 0.109877, "input_tokens": 800, "output_tokens": 500}
    },
    "markdown_file": "agentic_rag_md/20250127_143052.md",
    "rag_result_file": "rag_result/rag_result_20250127_143052.json",
    "process_steps": [...]
}
```

### 保存的文件

#### 1. Markdown文件 (`agentic_rag_md/`)
包含完整的处理过程记录：
- 📝 原始查询和意图识别
- 💰 费用统计
- ⏱️ 处理步骤统计
- 🔄 查询改写详情
- 🔍 检索结果详情
- 📋 最终答案

#### 2. JSON文件 (`rag_result/`)
包含完整的API检索数据：
- 元数据（时间戳、查询信息等）
- 费用分析
- 处理步骤详情
- 改写查询列表
- 检索结果详情（包括chunk检索和决策结果）

## 🔍 API接口说明

### KnowledgeRAG API
系统使用以下三个API接口：

1. **文档检索** (`/doc-retrieval-by-dataset`)
   - 获取文档summary和doc_ids
   - 输入：查询文本、数据集类型
   - 输出：文档ID列表和总结

2. **Chunk检索** (`/chunk-retrieval`)
   - 获取相关chunks
   - 输入：查询文本、文档ID列表
   - 输出：相关chunk列表

3. **Chunk决策** (`/chunk-decision`)
   - 对chunks进行决策匹配
   - 输入：查询文本、chunk列表
   - 输出：决策结果列表

## 💰 费用计算

### OpenAI API费用
- **输入费用**: $2.00 per 1M tokens
- **输出费用**: $8.00 per 1M tokens


## 📚 示例代码

### 完整示例
```python
#!/usr/bin/env python3
"""
复杂检索生成系统使用示例
"""

import asyncio
import os
from complex_retri_generate import ComplexRetriConfig, ComplexRetriWorkflow

async def example_usage():
    # 设置API密钥
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # 配置
    config = ComplexRetriConfig(
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_tokens=3000,
        max_queries=3
    )
    
    # 创建工作流
    workflow = ComplexRetriWorkflow(config)
    
    # 测试查询
    test_queries = [
        "What is a large language model?",
        "Please sort out all the professional terms in these papers for me",
        "Explain the differences between different training methods"
    ]
    
    print("=" * 80)
    print("复杂检索生成系统测试")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_queries)}: {query}")
        print(f"{'='*60}")
        
        try:
            result = await workflow.process(query)
            
            if result["success"]:
                print(f"✅ 处理成功")
                print(f"📊 意图类型: {'信息汇总型' if result['intent']['intent_type'] == 1 else '事实型'}")
                print(f"🔄 改写查询数: {result['rewritten_queries']}")
                print(f"🔍 检索成功率: {result['retrieval_success_rate']}")
                print(f"⏱️ 总耗时: {result['total_time']:.2f}秒")
                print(f"💰 总费用: ${result['total_cost_usd']:.6f} USD")
                
                # 显示各步骤费用
                print("\n💰 各步骤费用:")
                for step_name, cost_info in result['step_costs'].items():
                    step_cost = cost_info.get("total_cost_usd", 0)
                    print(f"  - {step_name}: ${step_cost:.6f}")
                
                # 显示最终答案的前200个字符
                answer_preview = result['final_answer'][:200] + "..." if len(result['final_answer']) > 200 else result['final_answer']
                print(f"\n📝 答案预览: {answer_preview}")
                
            else:
                print(f"❌ 处理失败: {result['error']}")
                
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
        
        print(f"\n{'='*60}")
    
    print("\n🎉 所有测试完成！")

if __name__ == "__main__":
    asyncio.run(example_usage())
```

**注意**: 使用前请确保已正确配置OpenAI API密钥和KnowledgeRAG API服务器地址。 