"""
Risk Analysis Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str, "objective": str, "options": str,
         "criteria": str, "constraints": str, "qa_context": str}
Output: RiskAnalysis
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import RiskAnalysis
from prompts.prompts import RISK_PROMPT


def get_risk_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(RiskAnalysis)
    chain = RISK_PROMPT | structured_llm
    return chain
