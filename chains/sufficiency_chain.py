"""
Information Sufficiency Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str, "objective": str, "options": str,
         "criteria": str, "constraints": str}
Output: SufficiencyCheck
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import SufficiencyCheck
from prompts.prompts import SUFFICIENCY_PROMPT


def get_sufficiency_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(SufficiencyCheck)
    chain = SUFFICIENCY_PROMPT | structured_llm
    return chain
