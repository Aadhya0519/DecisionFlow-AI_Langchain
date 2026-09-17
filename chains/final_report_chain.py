"""
Final Decision Report Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str, "category": str, "objective": str,
         "options": str, "criteria": str, "constraints": str,
         "tradeoffs": str, "risks": str, "uncertainties": str,
         "missing_info": str, "qa_context": str}
Output: FinalReport
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import FinalReport
from prompts.prompts import FINAL_REPORT_PROMPT


def get_final_report_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(FinalReport)
    chain = FINAL_REPORT_PROMPT | structured_llm
    return chain
