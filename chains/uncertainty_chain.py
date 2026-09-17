"""
Uncertainty Analysis Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str, "objective": str, "options": str,
         "criteria": str, "constraints": str, "qa_context": str}
Output: UncertaintyAnalysis
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import UncertaintyAnalysis
from prompts.prompts import UNCERTAINTY_PROMPT


def get_uncertainty_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(UncertaintyAnalysis)
    chain = UNCERTAINTY_PROMPT | structured_llm
    return chain
