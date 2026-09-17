"""
Constraint Extraction Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str}
Output: ConstraintList
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import ConstraintList
from prompts.prompts import CONSTRAINT_PROMPT


def get_constraint_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(ConstraintList)
    chain = CONSTRAINT_PROMPT | structured_llm
    return chain
