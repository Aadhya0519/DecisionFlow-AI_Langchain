"""
Option Extraction Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str}
Output: OptionList
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import OptionList
from prompts.prompts import OPTION_EXTRACTION_PROMPT


def get_option_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(OptionList)
    chain = OPTION_EXTRACTION_PROMPT | structured_llm
    return chain
