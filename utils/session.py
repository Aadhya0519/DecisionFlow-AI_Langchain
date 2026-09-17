"""
Helpers for managing Streamlit session state in DecisionFlow AI.

A "decision session" moves through three stages:

    "input"     -> user is typing/describing the decision
    "questions" -> system asked follow-up questions, waiting for answers
    "report"    -> final decision report is ready to view

Session state keys used across the app:

    stage           : str, one of "input" / "questions" / "report"
    decision_text   : str, the raw text the user entered
    understanding   : dict with "decision", "category", "objective"
    options         : list[str]
    criteria        : list[str]
    constraints     : list[str]
    missing_info    : list[str]
    questions       : list[str]
    qa_pairs        : list[tuple[str, str]]  (question, answer)
    tradeoffs       : list[dict]
    risks           : list[dict]
    uncertainties   : list[str]
    final_report    : dict (a FinalReport, dumped to a dict)
    history         : list[dict]  (past completed sessions)
"""

import streamlit as st

_DEFAULTS = {
    "stage": "input",
    "decision_text": "",
    "understanding": None,
    "options": [],
    "criteria": [],
    "constraints": [],
    "missing_info": [],
    "questions": [],
    "qa_pairs": [],
    "tradeoffs": [],
    "risks": [],
    "uncertainties": [],
    "final_report": None,
    "history": [],
}


def init_session_state() -> None:
    """Populate any missing session-state keys with their default values."""
    for key, default_value in _DEFAULTS.items():
        if key not in st.session_state:
            # Use a fresh copy for mutable defaults (lists) so sessions don't share state.
            st.session_state[key] = default_value.copy() if isinstance(default_value, list) else default_value


def start_new_decision() -> None:
    """Archive the current session (if a report exists) and reset for a new decision."""
    if st.session_state.get("final_report") is not None:
        st.session_state["history"].append(
            {
                "decision_text": st.session_state["decision_text"],
                "final_report": st.session_state["final_report"],
            }
        )

    for key, default_value in _DEFAULTS.items():
        if key == "history":
            continue  # keep history across resets
        st.session_state[key] = default_value.copy() if isinstance(default_value, list) else default_value
