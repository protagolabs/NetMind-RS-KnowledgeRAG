#!/usr/bin/env python3
"""
简化版 Deep Research 脚本

将复杂的 Next.js deep research 流程简化为 Python 脚本，适配 Document Retrieval API。
主要包含两个模块：
1. ResearchSteps: 分步骤研究操作
2. ResearchWorkflow: 完整的研究工作流

Author: yujing.wang
Date: 2025.01.27
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os
import logging
from pydantic import BaseModel
from typing import List, Optional
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """研究配置类。
    
    Attributes:
        max_depth: 最大研究深度
        time_limit_minutes: 时间限制（分钟）
        max_failed_attempts: 最大失败尝试次数
        model_name: 使用的AI模型名称
        temperature: AI模型温度参数
        max_tokens: 最大token数
    """
    max_depth: int = 5
    time_limit_minutes: int = 5
    max_failed_attempts: int = 3
    model_name: str = "gpt-4.1"
    temperature: float = 0.1
    max_tokens: int = 4000


@dataclass
class ResearchFinding:
    """研究发现的数据结构。
    
    Attributes:
        text: 发现内容文本
        source: 信息来源
        timestamp: 时间戳
        search_topic: 搜索主题
    """
    text: str
    source: str
    timestamp: str
    search_topic: str  # 添加搜索主题字段


@dataclass
class ResearchState:
    """研究状态管理。
    
    Attributes:
        findings: 研究发现列表
        summaries: 总结列表
        current_depth: 当前研究深度
        failed_attempts: 失败尝试次数
        max_failed_attempts: 最大失败尝试次数
        max_depth: 最大研究深度
    """
    findings: List[ResearchFinding] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    current_depth: int = 0
    failed_attempts: int = 0
    max_failed_attempts: int = 3
    max_depth: int = 5

class AnalysisResult(BaseModel):
    summary: str
    gaps: List[str]
    nextSteps: List[str]
    shouldContinue: bool
    nextSearchTopic: Optional[str] = None
    reasoning: Optional[str] = None

class ResearchSteps:
    """分步骤研究操作模块。
    
    提供独立的研究步骤功能，包括搜索、分析、生成报告等。
    每个方法都可以独立调用，便于调试和测试。
    """
    
    def __init__(self, rag_api_url: str = "http://localhost:8001", 
                 openai_api_key: Optional[str] = None, config: Optional[ResearchConfig] = None):
        """初始化研究步骤模块。
        
        Args:
            rag_api_url: 检索API的URL，默认使用端口8001
            openai_api_key: OpenAI API密钥，如果为None则从环境变量获取
            config: 研究配置，如果为None则使用默认配置
        """
        self.rag_api_url = rag_api_url
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.config = config or ResearchConfig()
        self.model_name = self.config.model_name
        self.client = openai.Client(api_key=self.openai_api_key)

        if not self.openai_api_key:
            raise ValueError("需要设置 OPENAI_API_KEY 环境变量")
    
    def _validate_analysis_result(self, analysis: AnalysisResult) -> bool:
        """验证分析结果的有效性。
        
        Args:
            analysis: 分析结果对象
            
        Returns:
            是否有效
        """
        if not analysis.summary or not analysis.summary.strip():
            logger.warning("分析结果缺少摘要")
            return False
        
        if not analysis.gaps:
            logger.warning("分析结果缺少知识缺口")
            return False
        
        if not analysis.nextSteps:
            logger.warning("分析结果缺少下一步建议")
            return False
        
        return True
    
    def _format_findings_for_analysis(self, findings: List[ResearchFinding]) -> str:
        """格式化研究发现用于分析。
        
        Args:
            findings: 研究发现列表
            
        Returns:
            格式化的文本
        """
        if not findings:
            return "暂无研究发现"
        
        # 按搜索主题分组
        findings_by_topic = {}
        for finding in findings:
            if finding.search_topic not in findings_by_topic:
                findings_by_topic[finding.search_topic] = []
            findings_by_topic[finding.search_topic].append(finding)
        
        formatted_sections = []
        for i, (search_topic, topic_findings) in enumerate(findings_by_topic.items(), 1):
            formatted_sections.append(f"## 搜索主题 {i}: {search_topic}")
            for j, finding in enumerate(topic_findings, 1):
                formatted_sections.append(f"[{i}.{j}] [{finding.source}]: {finding.text}")
            formatted_sections.append("")  # 空行分隔
        
        return "\n".join(formatted_sections)
    
    def _handle_openai_error(self, error: Exception, operation: str) -> None:
        """处理OpenAI API错误。
        
        Args:
            error: 错误对象
            operation: 操作名称
        """
        if "rate_limit" in str(error).lower():
            logger.error(f"{operation} - OpenAI API速率限制，请稍后重试")
        elif "quota" in str(error).lower():
            logger.error(f"{operation} - OpenAI API配额已用完")
        elif "authentication" in str(error).lower():
            logger.error(f"{operation} - OpenAI API认证失败，请检查API密钥")
        else:
            logger.error(f"{operation} - OpenAI API错误: {error}")
    
    def check_api_health(self) -> bool:
        """检查API健康状态。
        
        Returns:
            API是否健康可用
        """
        try:
            response = requests.get(f"{self.rag_api_url}/health", timeout=10)
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
                f"{self.rag_api_url}/search/simple",
                json={
                    "query": query,
                    "search_type": "hierarchical",
                    "model": self.model_name
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
    
    def get_detailed_search_results(self, query: str) -> Dict[str, Any]:
        """获取详细的搜索结果（包含摘要和chunk信息）。
        
        Args:
            query: 搜索查询
            
        Returns:
            详细搜索结果字典
        """
        try:
            logger.info(f"获取详细搜索结果: {query}")
            
            response = requests.post(
                f"{self.rag_api_url}/search",
                json={
                    "query": query,
                    "search_type": "hierarchical",
                    "model": self.model_name
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}"}
            
            result = response.json()
            
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "搜索失败")}
            
            logger.info("获取到详细搜索结果")
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"获取详细搜索结果错误: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_findings(self, topic, findings: List[ResearchFinding], 
                        time_remaining: float) -> Optional[AnalysisResult]:
        """分析发现并规划下一步。
        
        Args:
            topic: 研究主题
            findings: 研究发现列表
            time_remaining: 剩余时间（分钟）
            
        Returns:
            分析结果对象，包含summary、gaps、nextSteps等信息
        """
        try:
            logger.info("分析发现...")
            
            findings_text = self._format_findings_for_analysis(findings)
            
            # 统计搜索主题
            search_topics = list(set(finding.search_topic for finding in findings))
            search_history = "\n".join([f"- {topic}" for topic in search_topics])
            
            prompt = f"""You are a senior research analyst AI, tasked with synthesizing and analyzing the following research findings about the topic: "{topic}".

Current time remaining: {time_remaining:.1f} minutes.

Search history (topics already explored):
{search_history}

Research findings organized by search topics:
{findings_text}

Please perform the following analysis and provide your response in the exact JSON format specified:

1. **summary**: Summarize concisely and hierarchically what has been learned so far. Avoid mere listing, instead organize by sub-topics or key points.

2. **gaps**: Identify all knowledge gaps remaining, distinguishing between (a) missing information due to insufficient documents, and (b) open scientific questions or uncertainties. Provide as a list of strings.

3. **nextSteps**: Propose specific and actionable research steps for the next iteration, each with a brief justification of why it is necessary or promising. Provide as a list of strings.

4. **shouldContinue**: Boolean indicating whether to continue research. Set to false if time remaining is less than 1 minute, or you judge there is no more valuable information to be gained.

5. **nextSearchTopic**: The most promising search topic for the next iteration (optional string). This should be different from the topics already searched.

6. **reasoning**: Brief explanation for the shouldContinue decision and nextSearchTopic choice (optional string).

Please provide your analysis in the following JSON format:
{{
    "summary": "your summary here",
    "gaps": ["gap1", "gap2", ...],
    "nextSteps": ["step1", "step2", ...],
    "shouldContinue": true/false,
    "nextSearchTopic": "optional next search topic",
    "reasoning": "optional reasoning"
}}
"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            # 解析JSON响应
            content = response.choices[0].message.content
            if not content:
                logger.error("OpenAI返回空内容")
                return None
                
            try:
                # 解析JSON并创建AnalysisResult对象
                json_data = json.loads(content)
                analysis_result = AnalysisResult(**json_data)
                
                # 验证分析结果
                if not self._validate_analysis_result(analysis_result):
                    logger.warning("分析结果验证失败，但继续使用")
                
                return analysis_result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"解析分析结果失败: {e}")
                logger.error(f"原始内容: {content}")
                return None
                
        except openai.RateLimitError as e:
            self._handle_openai_error(e, "分析发现")
            return None
        except openai.AuthenticationError as e:
            self._handle_openai_error(e, "分析发现")
            return None
        except openai.APIError as e:
            self._handle_openai_error(e, "分析发现")
            return None
        except Exception as e:
            logger.error(f"分析错误: {e}")
            return None
    
    def generate_final_analysis(self, topic: str, findings: List[ResearchFinding], 
                              summaries: List[str]) -> str:
        """生成最终分析报告。
        
        Args:
            topic: 研究主题
            findings: 研究发现列表
            summaries: 总结列表
            
        Returns:
            最终分析报告文本
        """
        try:
            logger.info("生成最终分析...")
            
            findings_text = self._format_findings_for_analysis(findings)
            summaries_text = "\n".join([f"[总结{i+1}]: {s}" for i, s in enumerate(summaries)])
            
            prompt = f"""You are an expert research assistant.

Your task is to write a comprehensive, in-depth research report on the topic: "{topic}".

Below are all findings collected from local documents during the research process, along with step-by-step analysis summaries after each search and reasoning iteration.

## Findings
{findings_text}

## Analysis Summaries
{summaries_text}

Please carefully review all the above findings and analysis summaries. Your report must:
1. Provide a hierarchical, detailed summary of what was discovered, using specific evidence and referencing sources as [1], [2], etc.;
2. Extract and highlight key insights and trends, making clear which findings are robust and which are tentative;
3. Draw clear, well-supported conclusions, distinguishing between solid results and any remaining open questions;
4. Explicitly list remaining uncertainties, ambiguities, or areas needing further investigation, with explanations;
5. List all references used, mapping each [number] to its source (from the Findings section).

Be objective and logical, avoiding repetition. If the analysis summaries contain conflicting or uncertain information, discuss these cases and their implications.

**If you need to synthesize information, always cite the original source(s) from the Findings, not the summaries.**

Reply with a structured, readable report, using clear section headings.
"""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
        
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if not content:
                return "生成最终分析时出错：返回内容为空"
            
            return content
            
        except openai.RateLimitError as e:
            self._handle_openai_error(e, "生成最终分析")
            return f"生成最终分析时出错：API速率限制 - {e}"
        except openai.AuthenticationError as e:
            self._handle_openai_error(e, "生成最终分析")
            return f"生成最终分析时出错：API认证失败 - {e}"
        except openai.APIError as e:
            self._handle_openai_error(e, "生成最终分析")
            return f"生成最终分析时出错：API错误 - {e}"
        except Exception as e:
            logger.error(f"生成最终分析错误: {e}")
            return f"生成最终分析时出错: {e}"


class ResearchWorkflow:
    """完整的研究工作流模块。
    
    整合所有研究步骤，提供端到端的研究流程。
    包含状态管理、错误处理、进度跟踪等功能。
    """
    
    def __init__(self, rag_api_url: str = "http://localhost:8001", 
                 openai_api_key: Optional[str] = None, config: Optional[ResearchConfig] = None):
        """初始化研究工作流。
        
        Args:
            rag_api_url: 检索API的URL
            openai_api_key: OpenAI API密钥
            config: 研究配置
        """
        self.config = config or ResearchConfig()
        self.steps = ResearchSteps(rag_api_url, openai_api_key, self.config)
        self.rag_api_url = rag_api_url
        self.research_log = []  # 添加研究日志列表
    
    def _print_progress(self, depth: int, max_depth: int, remaining_time: float, 
                       findings_count: int, summaries_count: int):
        """打印研究进度。
        
        Args:
            depth: 当前深度
            max_depth: 最大深度
            remaining_time: 剩余时间
            findings_count: 发现数量
            summaries_count: 总结数量
        """
        print(f"\n{'='*60}")
        print(f"📊 研究进度: 深度 {depth}/{max_depth}")
        print(f"⏱️  剩余时间: {remaining_time:.1f} 分钟")
        print(f"📄 发现数量: {findings_count}")
        print(f"📋 总结数量: {summaries_count}")
        print(f"{'='*60}")
    
    def _log_step(self, step_info: Dict[str, Any]):
        """记录研究步骤到日志。
        
        Args:
            step_info: 步骤信息字典
        """
        self.research_log.append(step_info)
    
    def _print_step_details(self, step_info: Dict[str, Any]):
        """打印步骤详细信息。
        
        Args:
            step_info: 步骤信息字典
        """
        print(f"\n🔍 步骤 {step_info['step_number']} - 深度 {step_info['depth']}")
        print(f"📝 搜索主题: {step_info['search_topic']}")
        print(f"⏰ 时间戳: {step_info['timestamp']}")
        
        if step_info.get('search_success'):
            print(f"✅ 搜索成功")
            print(f"📄 发现内容: {step_info['finding_text'][:200]}...")
        else:
            print(f"❌ 搜索失败: {step_info.get('search_error', '未知错误')}")
        
        if step_info.get('analysis'):
            analysis = step_info['analysis']
            print(f"\n📋 分析结果:")
            print(f"   摘要: {analysis.summary[:150]}...")
            print(f"   知识缺口: {len(analysis.gaps)} 个")
            for i, gap in enumerate(analysis.gaps[:3], 1):  # 只显示前3个
                print(f"     {i}. {gap}")
            print(f"   下一步计划: {len(analysis.nextSteps)} 个")
            for i, step in enumerate(analysis.nextSteps[:3], 1):  # 只显示前3个
                print(f"     {i}. {step}")
            print(f"   是否继续: {'是' if analysis.shouldContinue else '否'}")
            if analysis.nextSearchTopic:
                print(f"   下一个搜索主题: {analysis.nextSearchTopic}")
            if analysis.reasoning:
                print(f"   推理: {analysis.reasoning}")
        
        print(f"{'='*60}")
    
    def _should_retry_operation(self, failed_attempts: int, max_attempts: int, 
                               error_type: str) -> bool:
        """判断是否应该重试操作。
        
        Args:
            failed_attempts: 失败次数
            max_attempts: 最大尝试次数
            error_type: 错误类型
            
        Returns:
            是否应该重试
        """
        if failed_attempts >= max_attempts:
            return False
        
        # 对于某些错误类型，不进行重试
        non_retryable_errors = ["authentication", "quota", "invalid_request"]
        if any(err in error_type.lower() for err in non_retryable_errors):
            return False
        
        return True
    
    def _save_markdown_result(self, topic: str, result: Dict[str, Any], 
                            research_log: List[Dict[str, Any]]) -> str:
        """保存研究结果为Markdown格式。
        
        Args:
            topic: 研究主题
            result: 研究结果
            research_log: 研究日志
            
        Returns:
            保存的文件路径
        """
        # 创建结果目录
        result_dir = "deep_research"
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.md"
        filepath = os.path.join(result_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 深度研究报告: {topic}\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 研究统计
            stats = result.get('stats', {})
            f.write("## 📊 研究统计\n\n")
            f.write(f"- **总用时**: {stats.get('total_time', 0):.1f} 秒\n")
            f.write(f"- **研究深度**: {stats.get('depth_reached', 0)}\n")
            f.write(f"- **发现数量**: {stats.get('findings_count', 0)}\n")
            f.write(f"- **总结数量**: {stats.get('summaries_count', 0)}\n\n")
            
            # 详细步骤记录
            f.write("## 🔍 研究步骤详情\n\n")
            for step in research_log:
                f.write(f"### 步骤 {step['step_number']} - 深度 {step['depth']}\n\n")
                f.write(f"**时间**: {step['timestamp']}\n\n")
                f.write(f"**搜索主题**: {step['search_topic']}\n\n")
                
                if step.get('search_success'):
                    f.write("**搜索状态**: ✅ 成功\n\n")
                    f.write("**发现内容**:\n")
                    f.write(f"```\n{step['finding_text']}\n```\n\n")
                else:
                    f.write(f"**搜索状态**: ❌ 失败 - {step.get('search_error', '未知错误')}\n\n")
                
                if step.get('analysis'):
                    analysis = step['analysis']
                    f.write("**分析结果**:\n\n")
                    f.write(f"**摘要**:\n{analysis.summary}\n\n")
                    
                    f.write("**知识缺口**:\n")
                    for i, gap in enumerate(analysis.gaps, 1):
                        f.write(f"{i}. {gap}\n")
                    f.write("\n")
                    
                    f.write("**下一步计划**:\n")
                    for i, next_step in enumerate(analysis.nextSteps, 1):
                        f.write(f"{i}. {next_step}\n")
                    f.write("\n")
                    
                    f.write(f"**是否继续**: {'是' if analysis.shouldContinue else '否'}\n\n")
                    
                    if analysis.nextSearchTopic:
                        f.write(f"**下一个搜索主题**: {analysis.nextSearchTopic}\n\n")
                    
                    if analysis.reasoning:
                        f.write(f"**推理**: {analysis.reasoning}\n\n")
                
                f.write("---\n\n")
            
            # 最终分析报告
            if result.get('final_analysis'):
                f.write("## 📋 最终分析报告\n\n")
                f.write(result['final_analysis'])
                f.write("\n\n")
            
            # 所有发现
            f.write("## 📄 所有研究发现\n\n")
            findings = result.get('findings', [])
            for i, finding in enumerate(findings, 1):
                f.write(f"### 发现 {i}\n\n")
                f.write(f"**搜索主题**: {finding['search_topic']}\n\n")
                f.write(f"**来源**: {finding['source']}\n\n")
                f.write(f"**时间**: {finding['timestamp']}\n\n")
                f.write(f"**内容**:\n```\n{finding['text']}\n```\n\n")
            
            # 所有总结
            f.write("## 📋 所有分析总结\n\n")
            summaries = result.get('summaries', [])
            for i, summary in enumerate(summaries, 1):
                f.write(f"### 总结 {i}\n\n")
                f.write(f"{summary}\n\n")
        
        return filepath
    
    def research(self, topic: str, max_depth: Optional[int] = None, 
                time_limit_minutes: Optional[int] = None) -> Dict[str, Any]:
        """执行完整的深度研究流程。
        
        Args:
            topic: 研究主题
            max_depth: 最大研究深度，如果为None则使用配置中的值
            time_limit_minutes: 时间限制（分钟），如果为None则使用配置中的值
            
        Returns:
            研究结果字典，包含success状态、findings、summaries、final_analysis等
        """
        # 使用配置值或传入的参数
        max_depth = max_depth or self.config.max_depth
        time_limit_minutes = time_limit_minutes or self.config.time_limit_minutes
        
        logger.info(f"开始深度研究: {topic}")
        logger.info(f"最大深度: {max_depth}, 时间限制: {time_limit_minutes} 分钟")
        logger.info(f"使用检索API: {self.rag_api_url}")
        print("=" * 60)
        
        # 检查API健康状态
        if not self.steps.check_api_health():
            return {
                "success": False,
                "error": "检索API不可用，请确保API服务器已启动",
                "topic": topic
            }
        
        start_time = time.time()
        time_limit = time_limit_minutes * 60  # 转换为秒
        
        # 初始化研究状态
        state = ResearchState(max_depth=max_depth, max_failed_attempts=self.config.max_failed_attempts)
        next_search_topic = topic
        step_number = 0
        
        try:
            while state.current_depth < max_depth:
                # 检查时间限制
                elapsed_time = time.time() - start_time
                if elapsed_time >= time_limit:
                    logger.info(f"时间限制已到 ({time_limit_minutes} 分钟)")
                    break
                
                state.current_depth += 1
                step_number += 1
                remaining_time = (time_limit - elapsed_time) / 60
                
                self._print_progress(state.current_depth, max_depth, remaining_time, 
                                    len(state.findings), len(state.summaries))
                
                # 搜索阶段
                search_topic = next_search_topic
                
                # 记录步骤开始
                step_info = {
                    'step_number': step_number,
                    'depth': state.current_depth,
                    'search_topic': search_topic,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'search_success': False,
                    'analysis': None
                }
                
                print(f"\n🔍 步骤 {step_number} - 深度 {state.current_depth}")
                print(f"📝 搜索主题: {search_topic}")
                
                search_result = self.steps.search_local_rag(search_topic)
                
                if not search_result["success"]:
                    logger.error(f"搜索失败: {search_result['error']}")
                    state.failed_attempts += 1
                    
                    # 更新步骤信息
                    step_info['search_error'] = search_result['error']
                    self._log_step(step_info)
                    self._print_step_details(step_info)
                    
                    if not self._should_retry_operation(state.failed_attempts, state.max_failed_attempts, search_result['error']):
                        logger.error("达到最大失败次数，停止研究")
                        break
                    continue
                
                # 处理搜索结果
                finding_text = "Here are related documents' summaries:\n\n" + \
                               search_result['data'].get("summary_results", "") + "\n" + \
                               "Here are related documents' chunks:\n\n" + \
                               search_result['data'].get("chunk_results", "")
                
                print(f"🔍 搜索结果: {finding_text[:500]}...")  # 打印前500个字符
                
                # 更新步骤信息
                step_info['search_success'] = True
                step_info['finding_text'] = finding_text
                
                # 添加发现
                finding = ResearchFinding(
                    text=finding_text,
                    source="local-documents",
                    timestamp=datetime.now().isoformat(),
                    search_topic=search_topic # 添加搜索主题
                )
                state.findings.append(finding)
                
                # 分析阶段
                print(f"\n📋 分析发现...")
                analysis = self.steps.analyze_findings(topic, state.findings, remaining_time)
                
                if not analysis:
                    logger.error("分析失败")
                    state.failed_attempts += 1
                    
                    # 更新步骤信息
                    step_info['analysis_error'] = "分析失败"
                    self._log_step(step_info)
                    self._print_step_details(step_info)
                    
                    if not self._should_retry_operation(state.failed_attempts, state.max_failed_attempts, "analysis_failed"):
                        logger.error("达到最大失败次数，停止研究")
                        break
                    continue
                
                # 更新步骤信息
                step_info['analysis'] = analysis
                self._log_step(step_info)
                self._print_step_details(step_info)
                
                # 添加分析总结
                if analysis.summary:
                    state.summaries.append(analysis.summary)
                
                # 检查是否继续研究
                if not analysis.shouldContinue:
                    logger.info("分析建议停止研究")
                    break
                
                # 检查是否还有知识缺口需要探索
                if not analysis.gaps:
                    logger.info("没有更多知识缺口需要探索")
                    break
                
                # 确定下一个搜索主题
                if analysis.nextSearchTopic:
                    next_search_topic = analysis.nextSearchTopic
                else:
                    # 如果没有明确指定下一个主题，使用第一个知识缺口
                    next_search_topic = analysis.gaps[0]
                
                logger.info(f"下一个搜索主题: {next_search_topic}")
            
            # 生成最终分析
            print(f"\n📋 生成最终分析...")
            final_analysis = self.steps.generate_final_analysis(
                topic, state.findings, state.summaries
            )
            
            total_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("🎉 研究完成!")
            print(f"⏱️  总用时: {total_time:.1f} 秒")
            print(f"📊 研究深度: {state.current_depth}")
            print(f"📄 发现数量: {len(state.findings)}")
            print(f"📋 总结数量: {len(state.summaries)}")
            print("=" * 60)
            
            result = {
                "success": True,
                "topic": topic,
                "findings": [{"text": f.text, "source": f.source, "timestamp": f.timestamp, "search_topic": f.search_topic} 
                           for f in state.findings],
                "summaries": state.summaries,
                "final_analysis": final_analysis,
                "stats": {
                    "total_time": total_time,
                    "depth_reached": state.current_depth,
                    "findings_count": len(state.findings),
                    "summaries_count": len(state.summaries)
                }
            }
            
            # 保存Markdown结果
            markdown_file = self._save_markdown_result(topic, result, self.research_log)
            print(f"📄 结果已保存到: {markdown_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"研究过程中出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }


def main():
    """主函数，执行研究并返回结果。
    
    这是一个简化的版本，直接执行研究并返回完整结果。
    """
    # 配置参数
    topic = "Please systematically summarize the latest research progress in the field of multimodal learning (Multimodal Learning / Multimodal AI)"
    
    # topic = "Please sort out all the professional terms in these papers for me."
    # 创建自定义配置
    config = ResearchConfig(
        max_depth=5,
        time_limit_minutes=5,
        max_failed_attempts=3,
        model_name="gpt-4.1",
        temperature=0.1,
        max_tokens=4000
    )
    
    rag_url = "http://localhost:8001"
    
    try:
        # 创建研究工作流实例
        workflow = ResearchWorkflow(
            rag_api_url=rag_url,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            config=config
        )
        
        # 执行研究
        result = workflow.research(topic=topic)
        
        # 返回完整结果
        return result
        
    except Exception as e:
        print(f"❌ 程序错误: {e}")
        return {
            "success": False,
            "error": str(e),
            "topic": topic
        }


def run_research(topic: str, max_depth: Optional[int] = None, 
                time_limit_minutes: Optional[int] = None, 
                rag_url: str = "http://localhost:8001",
                config: Optional[ResearchConfig] = None) -> Dict[str, Any]:
    """运行研究的简化函数。
    
    Args:
        topic: 要研究的主题或问题
        max_depth: 最大研究深度，如果为None则使用配置中的值
        time_limit_minutes: 时间限制(分钟)，如果为None则使用配置中的值
        rag_url: 检索API URL (默认: http://localhost:8001)
        config: 研究配置，如果为None则使用默认配置
        
    Returns:
        研究结果字典，包含success状态、findings、summaries、final_analysis等
    """
    try:
        # 创建研究工作流实例
        workflow = ResearchWorkflow(
            rag_api_url=rag_url,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            config=config
        )
        
        # 执行研究
        result = workflow.research(
            topic=topic,
            max_depth=max_depth,
            time_limit_minutes=time_limit_minutes
        )
        
        return result
        
    except Exception as e:
        logger.error(f"研究执行错误: {e}")
        return {
            "success": False,
            "error": str(e),
            "topic": topic
        }


# 使用示例
def example_usage():
    """使用示例函数。
    
    展示如何使用不同的配置进行深度研究。
    """
    print("🔍 Deep Research 使用示例")
    print("=" * 60)
    
    # 示例1: 使用默认配置
    print("\n📝 示例1: 使用默认配置")
    config1 = ResearchConfig()
    result1 = run_research(
        topic="人工智能在医疗领域的应用",
        config=config1
    )
    print(f"结果: {'成功' if result1['success'] else '失败'}")
    
    # 示例2: 自定义配置
    print("\n📝 示例2: 自定义配置")
    config2 = ResearchConfig(
        max_depth=2,
        time_limit_minutes=3,
        model_name="gpt-4o",
        temperature=0.2
    )
    result2 = run_research(
        topic="区块链技术发展趋势",
        config=config2
    )
    print(f"结果: {'成功' if result2['success'] else '失败'}")
    
    # 示例3: 直接使用工作流
    print("\n📝 示例3: 直接使用工作流")
    workflow = ResearchWorkflow(
        rag_api_url="http://localhost:8001",
        config=ResearchConfig(max_depth=1, time_limit_minutes=2)
    )
    result3 = workflow.research("机器学习算法比较")
    print(f"结果: {'成功' if result3['success'] else '失败'}")


# 配置说明
def print_config_help():
    """打印配置说明。"""
    print("🔧 配置说明")
    print("=" * 60)
    print("""
ResearchConfig 参数说明:
- max_depth: 最大研究深度 (默认: 5)
- time_limit_minutes: 时间限制，分钟 (默认: 5)
- max_failed_attempts: 最大失败尝试次数 (默认: 3)
- model_name: OpenAI模型名称 (默认: "gpt-4o")
- temperature: AI模型温度参数 (默认: 0.1)
- max_tokens: 最大token数 (默认: 4000)

使用建议:
1. 对于快速探索，设置 max_depth=2, time_limit_minutes=3
2. 对于深度研究，设置 max_depth=5, time_limit_minutes=10
3. 对于创意性研究，可以增加 temperature 到 0.3-0.5
4. 对于技术性研究，保持 temperature=0.1 以获得更稳定的结果
    """)


if __name__ == "__main__":
    # 如果需要查看配置说明，取消下面的注释
    # print_config_help()
    
    # 如果需要运行示例，取消下面的注释
    # example_usage()
    
    # 直接执行研究并打印结果
    result = main()
    
    if result.get("success"):
        print("\n" + "=" * 60)
        print("📋 最终分析报告")
        print("=" * 60)
        print(result.get("final_analysis", "无分析结果"))
        
        # 打印统计信息
        stats = result.get("stats", {})
        print(f"\n📊 研究统计:")
        print(f"   ⏱️  总用时: {stats.get('total_time', 0):.1f} 秒")
        print(f"   📊 研究深度: {stats.get('depth_reached', 0)}")
        print(f"   📄 发现数量: {stats.get('findings_count', 0)}")
        print(f"   📋 总结数量: {stats.get('summaries_count', 0)}")
    else:
        print(f"❌ 研究失败: {result.get('error', '未知错误')}") 