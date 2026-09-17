"""
Trade-off Analysis Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str, "objective": str, "options": str,
         "criteria": str, "constraints": str, "qa_context": str}
Output: TradeoffAnalysis
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import TradeoffAnalysis
from prompts.prompts import TRADEOFF_PROMPT


def get_tradeoff_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(TradeoffAnalysis)
    chain = TRADEOFF_PROMPT | structured_llm
    return chain
