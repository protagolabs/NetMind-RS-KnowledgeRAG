#!/usr/bin/env python3
"""
调试版 Deep Research 脚本
显示详细的中间过程
"""

import os
import json
from simple_deep_research import run_research

def debug_research():
    """调试版研究，显示详细过程"""
    
    print("=== 调试版深度研究 ===")
    print("研究主题: What does memory mean in LLM agents?")
    print("最大深度: 2")
    print("时间限制: 3 分钟")
    print("=" * 50)
    
    # 执行研究
    result = run_research(
        topic="What does memory mean in LLM agents?",
        max_depth=2,
        time_limit_minutes=3
    )
    
    # 详细分析结果
    if result.get("success"):
        print("\n✅ 研究成功完成!")
        
        # 显示所有发现
        findings = result.get("findings", [])
        print(f"\n📄 所有发现 ({len(findings)} 个):")
        for i, finding in enumerate(findings, 1):
            print(f"\n发现 {i}:")
            print(f"  来源: {finding['source']}")
            print(f"  时间: {finding['timestamp']}")
            print(f"  内容: {finding['text']}")
        
        # 显示所有总结
        summaries = result.get("summaries", [])
        print(f"\n📋 所有总结 ({len(summaries)} 个):")
        for i, summary in enumerate(summaries, 1):
            print(f"\n总结 {i}:")
            print(f"  内容: {summary}")
        
        # 显示最终分析
        final_analysis = result.get("final_analysis", "")
        print(f"\n📊 最终分析报告:")
        print("=" * 60)
        print(final_analysis)
        
        # 显示统计信息
        stats = result.get("stats", {})
        print(f"\n📈 研究统计:")
        print(f"   ⏱️  总用时: {stats.get('total_time', 0):.1f} 秒")
        print(f"   📊 研究深度: {stats.get('depth_reached', 0)}")
        print(f"   📄 发现数量: {stats.get('findings_count', 0)}")
        print(f"   📋 总结数量: {stats.get('summaries_count', 0)}")
        
        # 保存完整结果
        with open("debug_research_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 完整调试结果已保存到: debug_research_result.json")
        
    else:
        print(f"\n❌ 研究失败: {result.get('error', '未知错误')}")
        
        # 即使失败也保存结果
        with open("debug_research_error.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 错误信息已保存到: debug_research_error.json")


if __name__ == "__main__":
    debug_research() 