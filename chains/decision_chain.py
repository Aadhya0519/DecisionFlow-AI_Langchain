"""
Decision Understanding Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str}
Output: DecisionUnderstanding
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import DecisionUnderstanding
from prompts.prompts import DECISION_UNDERSTANDING_PROMPT


def get_decision_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(DecisionUnderstanding)
    chain = DECISION_UNDERSTANDING_PROMPT | structured_llm
    return chain
