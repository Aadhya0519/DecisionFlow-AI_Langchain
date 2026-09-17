"""
Criteria Identification Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str}
Output: CriteriaList
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import CriteriaList
from prompts.prompts import CRITERIA_PROMPT


def get_criteria_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(CriteriaList)
    chain = CRITERIA_PROMPT | structured_llm
    return chain
