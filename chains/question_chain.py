"""
Follow-up Question Chain.

LCEL pipeline: ChatPromptTemplate | structured-output LLM
Input : {"decision_text": str, "objective": str, "missing_info": str}
Output: FollowUpQuestions
"""

from langchain_core.runnables import RunnableSequence
from models.schemas import FollowUpQuestions
from prompts.prompts import QUESTION_PROMPT


def get_question_chain(llm) -> RunnableSequence:
    structured_llm = llm.with_structured_output(FollowUpQuestions)
    chain = QUESTION_PROMPT | structured_llm
    return chain
