"""
@file_name: step_2_writing.py
@author: bin.liang
@date: 2025-08-01
@description: This file is used to write the answer to the user question.
"""


from copy import deepcopy

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from pydantic import BaseModel

from knowledge_rag.config import OPENAI_API_KEY


class Plan(BaseModel):
    reasoning: str
    plan: str


WRITING_PROMPT = """ 
You are a good scientist, you are answering a question about the changes in model training methods.

## You have the following information:
{information}      

## Your plan is:
{plan}

## The user question is:
{user_question}

## Your target:
Following the plan to write a answer to the user question.

## Requirements:
- Each insights/ideas/analysis should have a reference to the source document.
- You can not say anything that is not in the information.
- You must use the special format to note the reference in each sentence, by use <reference>{{the order of the reference}}</reference>
- Your answer must be very detailed and comprehensive.
- You must be professional and concise.
"""


async def make_rag_writing(information: str, user_question: str, plan: str):
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    local_writing_prompt = deepcopy(WRITING_PROMPT)
    writing_prompt = local_writing_prompt.format(
        information=information, 
        plan=plan,
        user_question=user_question)

    writing_agent = Agent(
        name="writing_agent",
        instructions=writing_prompt,
        model=OpenAIChatCompletionsModel(
            model="gpt-4.1",
            openai_client=client,
        ),
    )

    runner = await Runner.run(
        writing_agent,
        "Please help me to write a answer to the user question by following the plan."
    )

    return runner.final_output




