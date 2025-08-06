"""
@file_name: step_1_planning.py
@author: bin.liang
@date: 2025-08-01
@description: This file is used to make a plan to answer the user question.
"""


from copy import deepcopy

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from pydantic import BaseModel

from knowledge_rag.config import OPENAI_API_KEY


class Plan(BaseModel):
    reasoning: str
    plan: str


PLAN_PROMPT = """
You are a good scientist, you are answering a question about the changes
in model training methods.

## You have the following information:
{information}

## The user question is:
{user_question}

## Your target:
Making a plan to use the information to answer the user question.
"""


async def make_rag_plan(information: str, user_question: str):
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    local_plan_prompt = deepcopy(PLAN_PROMPT)
    plan_prompt = local_plan_prompt.format(
        information=information, user_question=user_question
    )

    plan_agent = Agent(
        name="plan_agent",
        instructions=plan_prompt,
        model=OpenAIChatCompletionsModel(
            model="gpt-4.1",
            openai_client=client,
        ),
        output_type=Plan,
    )

    runner = await Runner.run(
        plan_agent,
        "Please help me to make a plan to answer the user question."
    )

    return runner.final_output



