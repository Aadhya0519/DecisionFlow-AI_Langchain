"""
Pipeline orchestration for DecisionFlow AI.

This module wires the 10 individual chains together into the workflow
described in the project spec:

    User Decision
        -> Understanding Stage        (RunnableParallel: decision/option/criteria/constraint chains)
        -> Sufficiency Stage          (sufficiency chain)
        -> Question Stage             (RunnableBranch: question chain OR empty list)
        -> Analysis Stage             (RunnableParallel: trade-off/risk/uncertainty chains)
        -> Final Report Stage         (final report chain)

Because step 3 (asking the user follow-up questions and waiting for their
answers) requires a real person in the loop, the workflow is exposed as a
handful of plain functions that the Streamlit app calls one at a time,
rather than as a single `.invoke()` spanning the whole conversation. Each
function internally still uses LCEL Runnables (RunnableParallel /
RunnableBranch) to do its part of the work.
"""

from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

from chains.decision_chain import get_decision_chain
from chains.option_chain import get_option_chain
from chains.criteria_chain import get_criteria_chain
from chains.constraint_chain import get_constraint_chain
from chains.sufficiency_chain import get_sufficiency_chain
from chains.question_chain import get_question_chain
from chains.tradeoff_chain import get_tradeoff_chain
from chains.risk_chain import get_risk_chain
from chains.uncertainty_chain import get_uncertainty_chain
from chains.final_report_chain import get_final_report_chain
from models.schemas import FollowUpQuestions


def _format_list(items) -> str:
    """Turn a list of strings into a readable bullet-point block for prompts."""
    if not items:
        return "None provided."
    return "\n".join(f"- {item}" for item in items)


def _format_qa_context(qa_pairs) -> str:
    """Turn a list of (question, answer) tuples into a readable block for prompts."""
    if not qa_pairs:
        return "No follow-up questions were needed."
    lines = [f"Q: {question}\nA: {answer}" for question, answer in qa_pairs]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1: Decision Understanding + Option + Criteria + Constraint
# (independent analyses -> run with RunnableParallel)
# ---------------------------------------------------------------------------
def run_understanding_stage(llm, decision_text: str) -> dict:
    understanding_stage = RunnableParallel(
        understanding=get_decision_chain(llm),
        option_result=get_option_chain(llm),
        criteria_result=get_criteria_chain(llm),
        constraint_result=get_constraint_chain(llm),
    )

    result = understanding_stage.invoke({"decision_text": decision_text})

    understanding = result["understanding"]
    options = result["option_result"].options
    criteria = result["criteria_result"].criteria
    constraints = result["constraint_result"].constraints

    return {
        "decision": understanding.decision,
        "category": understanding.category,
        "objective": understanding.objective,
        "options": options,
        "criteria": criteria,
        "constraints": constraints,
    }


# ---------------------------------------------------------------------------
# Stage 2 + 3: Sufficiency check, then branch to follow-up questions
# (RunnableBranch: insufficient -> question chain, sufficient -> no questions)
# ---------------------------------------------------------------------------
def run_sufficiency_and_question_stage(llm, context: dict) -> dict:
    sufficiency_chain = get_sufficiency_chain(llm)
    question_chain = get_question_chain(llm)

    sufficiency_input = {
        "decision_text": context["decision"],
        "objective": context["objective"],
        "options": _format_list(context["options"]),
        "criteria": _format_list(context["criteria"]),
        "constraints": _format_list(context["constraints"]),
    }
    sufficiency_result = sufficiency_chain.invoke(sufficiency_input)

    def needs_questions(_input: dict) -> bool:
        return not sufficiency_result.is_sufficient

    # RunnableBranch: route to the question chain only when information is
    # insufficient; otherwise pass through an empty FollowUpQuestions.
    question_router = RunnableBranch(
        (needs_questions, question_chain),
        RunnableLambda(lambda _input: FollowUpQuestions(questions=[])),
    )

    question_input = {
        "decision_text": context["decision"],
        "objective": context["objective"],
        "missing_info": _format_list(sufficiency_result.missing_info),
    }
    question_result = question_router.invoke(question_input)

    return {
        "is_sufficient": sufficiency_result.is_sufficient,
        "missing_info": sufficiency_result.missing_info,
        "reasoning": sufficiency_result.reasoning,
        "questions": question_result.questions,
    }


# ---------------------------------------------------------------------------
# Stage 4: Trade-off + Risk + Uncertainty analysis
# (independent analyses -> run with RunnableParallel)
# ---------------------------------------------------------------------------
def run_analysis_stage(llm, context: dict, qa_pairs) -> dict:
    analysis_stage = RunnableParallel(
        tradeoff_result=get_tradeoff_chain(llm),
        risk_result=get_risk_chain(llm),
        uncertainty_result=get_uncertainty_chain(llm),
    )

    analysis_input = {
        "decision_text": context["decision"],
        "objective": context["objective"],
        "options": _format_list(context["options"]),
        "criteria": _format_list(context["criteria"]),
        "constraints": _format_list(context["constraints"]),
        "qa_context": _format_qa_context(qa_pairs),
    }

    result = analysis_stage.invoke(analysis_input)

    tradeoffs = [t.model_dump() for t in result["tradeoff_result"].tradeoffs]
    risks = [r.model_dump() for r in result["risk_result"].risk_analysis]
    uncertainties = result["uncertainty_result"].uncertainties

    return {
        "tradeoffs": tradeoffs,
        "risks": risks,
        "uncertainties": uncertainties,
    }


# ---------------------------------------------------------------------------
# Stage 5: Final Decision Report
# ---------------------------------------------------------------------------
def run_final_report_stage(llm, context: dict, analysis: dict, missing_info, qa_pairs) -> dict:
    final_report_chain = get_final_report_chain(llm)

    final_input = {
        "decision_text": context["decision"],
        "category": context["category"],
        "objective": context["objective"],
        "options": _format_list(context["options"]),
        "criteria": _format_list(context["criteria"]),
        "constraints": _format_list(context["constraints"]),
        "tradeoffs": _format_list(
            [f"{t['option']}: gains {t['gains']}, sacrifices {t['sacrifices']}" for t in analysis["tradeoffs"]]
        ),
        "risks": _format_list([f"{r['option']}: {r['risks']}" for r in analysis["risks"]]),
        "uncertainties": _format_list(analysis["uncertainties"]),
        "missing_info": _format_list(missing_info),
        "qa_context": _format_qa_context(qa_pairs),
    }

    final_report = final_report_chain.invoke(final_input)
    return final_report.model_dump()
