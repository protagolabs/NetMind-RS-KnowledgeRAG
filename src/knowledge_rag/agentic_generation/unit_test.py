""" 
@file_name: unit_test.py
@author: Yujing Wang, Bin Liang
@date: 2025-08-11
@description: 
    RAG智能代理单元测试模块
    
    本模块提供了RAG智能代理系统的基本测试用例，用于验证系统的
    核心功能是否正常工作。包括端到端的查询处理测试。
    
    测试内容：
    1. RAG代理初始化测试
    2. 查询处理流程测试
    3. 结果输出验证
    
    使用方法：
        python unit_test.py
    
    测试查询示例：
        - "Please sort out all the professional terms in these papers for me"
        - 这是一个信息汇总型查询，用于测试系统的术语提取和整理能力
"""

import json
from tqdm.auto import tqdm
from typing import List
from knowledge_rag.agentic_generation.rag_agent import RAGAgent


async def main():
    """主函数，使用异步上下文管理器确保资源正确释放"""
    
    # with open("data/questions.json", "r") as f:
    #     questions = json.load(f)
    # query = "For the Patriot Exchange policy's Accidental Death & Dismemberment benefit, what is the maximum principal sum payable, and name two specific types of activities or conditions explicitly listed as exclusions for accidental death or dismemberment coverage? "
    # print(query)
    # async with RAGAgent() as rag_agent:
    #     result = await rag_agent.rag_agent(query, dataset_type="baoxian") 
    # print(f"Chunk ids: {result['chunk_ids']}")
    # print(f"Cost breakdown: {result['cost_breakdown']}")
    # print(f"Performance metrics: {result['performance_metrics']}")
    # print(f"Metadata: {result['metadata']}")
    
    # questions =[
    #     "What is the Important Notice and Disclaimer Concerning the United States Patient Protection and Affordable Care Act (PPACA) for these insurance plans, and what implications does it have for U.S. citizens or residents?",
    #     "How are pre-existing conditions generally handled across the Atlas Travel, Atlas, StudentSecure, and GeoBlue Navigator plans, and what are the specific conditions that qualify for coverage under the 'Acute Onset of Pre-existing Conditions' benefit?",
    #     "What are the eligibility requirements for members applying for coverage under the Atlas Travel, StudentSecure, GeoBlue Navigator, and Patriot Exchange plans, including any age or visa-related criteria?",
    #     "What are the rules and associated fees for cancelling a policy and receiving a refund for Atlas Travel, StudentSecure, and Patriot Exchange plans?",
    #     "How do the Atlas Travel, StudentSecure, and Patriot Exchange plans utilize a U.S. Preferred Provider Organization (PPO) network for medical treatment within the United States, and what are the benefits of using PPO providers?",
    #     "Outline the steps and submission timelines required to file a claim for medical expenses under the Atlas Travel, Atlas, StudentSecure, and Patriot Exchange plans.",
    #     "Describe the Appeals and Complaints Procedure available to policyholders of Atlas Travel, StudentSecure, and GeoBlue Navigator if a claim is denied or if they wish to make a complaint.",
    #     "Explain the Arbitration and Class Action Waiver clause in the Atlas Travel and StudentSecure policies, including the types of disputes it covers, the arbitration process, and how a member can opt out.",
    #     "What are the overall maximum limits for medical expenses in the various Atlas Travel, Atlas, and StudentSecure plans, and how do these limits vary based on the insured person's age or the specific plan type?",
    #     "Detail the emergency room co-payment requirements for claims incurred both in the U.S. and outside the U.S. under the Atlas Travel, Atlas, and StudentSecure plans.",
    #     "What specific sports and activities are generally excluded from coverage across the Atlas Travel, StudentSecure, and Patriot Exchange plans, and are there any optional riders available for additional coverage for certain activities?",
    #     "What are the conditions and limitations for eligible medical expenses resulting from an Act of Terrorism under Atlas Travel, StudentSecure, and Patriot Exchange, including geographic restrictions and types of agents used?",
    #     "How is mental health disorder treatment covered under the StudentSecure plans (Elite, Select, Budget, Smart), what are the limits on visits or inpatient days, and what are the specific exclusions for mental health services across StudentSecure and GeoBlue Navigator?",
    #     "Compare the maternity care coverage, including complications of pregnancy and newborn care, as described in the StudentSecure and GeoBlue Navigator plans, and highlight any differences or exclusions, particularly with the Patriot Exchange plan.",
    #     "Under the Patriot Exchange plan, which services require pre-certification, and what are the financial consequences (e.g., reduction of eligible medical expenses, maximum penalties) if these pre-certification requirements are not met?",
    #     "Provide the definition of 'Pre-existing Condition' as found in the Atlas Travel, Atlas, StudentSecure, and GeoBlue Navigator documents, noting any variations in the look-back periods.",
    #     "What are the rules regarding eligibility for coverage within or outside of the home country for U.S. citizens and non-U.S. citizens under the Atlas Travel and StudentSecure plans, including restrictions on establishing a new home country?",
    #     "How do deductibles and coinsurance maximums cross-accumulate or apply across U.S. Participating Provider, U.S. Non-Participating Provider, and International locations within the GeoBlue Navigator plan?",
    #     "Describe the benefits and exclusions for Accidental Death & Dismemberment under the Atlas Travel, StudentSecure, and GeoBlue Navigator plans, including any age-based limits or specific causes of loss that are not covered.",
    #     "What coverage is offered for natural disasters, specifically for replacement accommodations and evacuation, under the Atlas Travel plan, including any optional riders like the Optional Crisis Response Benefit Rider?"
    # ]
    
    import json 
    with open("experiments/apple_structured_questions.json", "r") as f:
        questions = json.load(f)
        
    questions = questions[:3]
    
    # questions = [
    #    #  "In what ways has the iPad product line evolved since its inception in 2010, and how have key updates or new models impacted Apple's market share and revenue through 2024?",
    #    #  "How has Apple's revenue from services like the App Store, iCloud, and Apple Music grown since their respective launches, and what role have they played in Apple's overall revenue composition through 2024?",
    #    #  "In what ways did the leadership transition from Steve Jobs to Tim Cook impact the strategic direction and financial performance of Apple between 2011 and 2024, particularly regarding product innovation and market expansion?",
    # ]
    
    all_results = []
    async with RAGAgent() as rag_agent:
        for question in tqdm(questions):
            query = question["question"]
            local_result = {
                "original_answer": question["answer"],
                "original_reference": question["reference"],
                "question_type": question["question_type"],
                "doc_id": question["doc_id"],
            }
            result = await rag_agent.rag_agent(query, dataset_type="apple") 
            local_result['xyz_answer'] = result['answer']
            local_result['chunks'] = result['chunk_ids']
            local_result['cost_breakdown'] = result['cost_breakdown']
            local_result['performance_metrics'] = result['performance_metrics']
            local_result['metadata'] = result['metadata']
            all_results.append(local_result)
            
            with open("data/results_20250828_4.json", "w") as f:
                json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    