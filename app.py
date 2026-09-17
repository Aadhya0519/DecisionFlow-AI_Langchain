"""
DecisionFlow AI - Intelligent Decision Analysis Engine
Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st

from config.settings import APP_NAME, APP_TAGLINE, EXAMPLE_DECISIONS
from utils.llm import get_llm
from utils.session import init_session_state, start_new_decision
from chains.pipeline import (
    run_understanding_stage,
    run_sufficiency_and_question_stage,
    run_analysis_stage,
    run_final_report_stage,
)

# Color accents for category chips - purely presentational, no functional impact.
CATEGORY_COLORS = {
    "Career": "#4F86F7",
    "Education": "#00B894",
    "Finance": "#F0932B",
    "Technology": "#A29BFE",
    "Purchase": "#FF6B6B",
    "Business": "#6C5CE7",
    "Lifestyle": "#FD79A8",
    "Other": "#8395A7",
}


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --df-primary: #6C5CE7;
        --df-primary-dark: #5844D6;
        --df-pink: #EC4899;
        --df-bg: #F6F7FC;
        --df-card: #FFFFFF;
        --df-text: #1E1B2E;
        --df-muted: #6C7293;
        --df-blue: #4F86F7;
        --df-green: #00B894;
        --df-orange: #F0932B;
        --df-red: #FF6B6B;
        --df-purple: #A29BFE;
        --df-amber: #FDCB6E;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--df-text);
    }
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: var(--df-text);
    }
    [data-testid="stAppViewContainer"], .stApp, [data-testid="stMain"] {
        background-color: var(--df-bg);
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #EDEBFB;
    }

    /* ---------- hero banner ---------- */
    .df-hero {
        background: linear-gradient(135deg, #6C5CE7 0%, #9B6DF0 45%, #EC4899 100%);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.25);
    }
    .df-hero h1 {
        color: #FFFFFF;
        font-size: 2.1rem;
        margin: 0;
    }
    .df-hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin: 0.35rem 0 0 0;
    }
    .df-sidebrand h2 {
        font-size: 1.3rem;
        margin: 0;
        background: linear-gradient(135deg, #6C5CE7, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .df-sidebrand p {
        color: var(--df-muted);
        font-size: 0.88rem;
        margin: 0.2rem 0 0 0;
    }

    /* ---------- cards ---------- */
    .df-card {
        background-color: var(--df-card);
        border-radius: 16px;
        border: 1px solid #EDEBFB;
        border-left: 5px solid var(--df-primary);
        box-shadow: 0 4px 18px rgba(108, 92, 231, 0.08);
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .df-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(108, 92, 231, 0.14);
    }
    .df-card h4 {
        margin-top: 0;
        font-size: 0.95rem;
        color: var(--df-primary);
    }
    .df-card ul { margin: 0.2rem 0 0.9rem 1.1rem; }
    .df-card ul:last-child { margin-bottom: 0; }
    .df-card li { margin-bottom: 0.3rem; line-height: 1.5; }
    .df-accent-blue   { border-left-color: var(--df-blue); }
    .df-accent-green  { border-left-color: var(--df-green); }
    .df-accent-orange { border-left-color: var(--df-orange); }
    .df-accent-red    { border-left-color: var(--df-red); }
    .df-accent-purple { border-left-color: var(--df-purple); }
    .df-accent-amber  { border-left-color: var(--df-amber); }

    .df-chip {
        display: inline-block;
        color: #FFFFFF;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
    }

    .df-callout {
        background: linear-gradient(135deg, #F1EEFE 0%, #FDEFF6 100%);
        border: 1px solid #E8E1FB;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        font-size: 1.02rem;
        line-height: 1.6;
        color: var(--df-text);
    }

    /* ---------- buttons ---------- */
    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #E0DCF9;
        color: var(--df-primary);
        background-color: #FFFFFF;
        transition: all 0.15s ease;
    }
    div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--df-primary);
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.18);
        transform: translateY(-1px);
    }
    div[data-testid="stButton"] button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #6C5CE7, #EC4899);
        border: none;
        color: #FFFFFF;
        box-shadow: 0 6px 18px rgba(108, 92, 231, 0.3);
    }
    div[data-testid="stButton"] button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        box-shadow: 0 8px 22px rgba(108, 92, 231, 0.4);
        transform: translateY(-1px);
    }

    /* ---------- inputs ---------- */
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
        border-radius: 10px;
        border: 1px solid #E0DCF9;
        background-color: #FFFFFF;
    }
    [data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus {
        border-color: var(--df-primary);
        box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.15);
    }

    /* ---------- tabs ---------- */
    [data-testid="stTabs"] button[role="tab"] {
        font-weight: 600;
        color: var(--df-muted);
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--df-primary);
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--df-primary);
        height: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_session_state()


@st.cache_resource
def load_llm():
    return get_llm()


def bullet_list(items, empty_text="None identified"):
    if not items:
        return f"<li>{empty_text}</li>"
    return "".join(f"<li>{item}</li>" for item in items)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""<div class="df-sidebrand"><h2>🧭 {APP_NAME}</h2><p>{APP_TAGLINE}</p></div>""",
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("➕ New Decision", use_container_width=True):
        start_new_decision()
        st.rerun()

    st.divider()
    st.markdown("### Session History")
    if not st.session_state["history"]:
        st.caption("Past decisions in this session will appear here.")
    else:
        for i, past in enumerate(reversed(st.session_state["history"]), start=1):
            with st.expander(past["decision_text"][:60] or f"Decision {i}"):
                st.write(past["final_report"]["neutral_summary"])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="df-hero"><h1>🧭 {APP_NAME}</h1><p>{APP_TAGLINE}</p></div>""",
    unsafe_allow_html=True,
)


def render_decision_canvas(context: dict) -> None:
    st.markdown("### 🗂️ Decision Canvas")
    chip_color = CATEGORY_COLORS.get(context["category"], "#8395A7")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""<div class="df-card">
            <h4>Decision</h4><p>{context['decision']}</p>
            <h4>Category</h4><p><span class="df-chip" style="background-color:{chip_color};">{context['category']}</span></p>
            <h4>Objective</h4><p>{context['objective']}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        options_html = bullet_list(context["options"])
        criteria_html = bullet_list(context["criteria"])
        constraints_html = bullet_list(context["constraints"])
        st.markdown(
            f"""<div class="df-card df-accent-blue">
            <h4>Options</h4><ul>{options_html}</ul>
            <h4>Criteria</h4><ul>{criteria_html}</ul>
            <h4>Constraints</h4><ul>{constraints_html}</ul>
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Stage: INPUT
# ---------------------------------------------------------------------------
if st.session_state["stage"] == "input":
    st.subheader("Describe the decision you're facing")

    decision_text = st.text_area(
        label="decision_input",
        label_visibility="collapsed",
        placeholder="Describe the decision you're facing...",
        height=140,
        value=st.session_state["decision_text"],
    )

    st.caption("Or try an example:")
    example_cols = st.columns(3)
    for i, example in enumerate(EXAMPLE_DECISIONS):
        if example_cols[i % 3].button(example, use_container_width=True, key=f"example_{i}"):
            decision_text = example
            st.session_state["decision_text"] = example
            st.rerun()

    analyze_clicked = st.button("Analyze Decision →", type="primary", use_container_width=True)

    if analyze_clicked:
        if not decision_text or not decision_text.strip():
            st.warning("Please describe a decision before analyzing.")
        else:
            st.session_state["decision_text"] = decision_text.strip()
            try:
                llm = load_llm()
                with st.spinner("Understanding your decision..."):
                    context = run_understanding_stage(llm, st.session_state["decision_text"])
                    st.session_state["understanding"] = {
                        "decision": context["decision"],
                        "category": context["category"],
                        "objective": context["objective"],
                    }
                    st.session_state["options"] = context["options"]
                    st.session_state["criteria"] = context["criteria"]
                    st.session_state["constraints"] = context["constraints"]

                with st.spinner("Checking whether enough information is available..."):
                    suff_result = run_sufficiency_and_question_stage(llm, context)
                    st.session_state["missing_info"] = suff_result["missing_info"]
                    st.session_state["questions"] = suff_result["questions"]

                if suff_result["is_sufficient"] or not suff_result["questions"]:
                    st.session_state["stage"] = "report"
                else:
                    st.session_state["stage"] = "questions"
                st.rerun()

            except Exception as e:
                st.error(f"Something went wrong while analyzing your decision: {e}")


# ---------------------------------------------------------------------------
# Stage: QUESTIONS
# ---------------------------------------------------------------------------
elif st.session_state["stage"] == "questions":
    context = {
        "decision": st.session_state["understanding"]["decision"],
        "category": st.session_state["understanding"]["category"],
        "objective": st.session_state["understanding"]["objective"],
        "options": st.session_state["options"],
        "criteria": st.session_state["criteria"],
        "constraints": st.session_state["constraints"],
    }
    render_decision_canvas(context)

    st.markdown("### ❓ A few quick questions")
    st.caption("Your answers help produce a more accurate analysis. You can also skip and analyze with what's available.")

    with st.form("followup_form"):
        answers = []
        for i, question in enumerate(st.session_state["questions"]):
            answer = st.text_input(question, key=f"question_{i}")
            answers.append(answer)

        col_a, col_b = st.columns(2)
        submitted = col_a.form_submit_button("Continue Analysis →", type="primary", use_container_width=True)
        skipped = col_b.form_submit_button("Skip and Analyze with Available Information", use_container_width=True)

    if submitted or skipped:
        qa_pairs = []
        if submitted:
            qa_pairs = [
                (q, a.strip()) for q, a in zip(st.session_state["questions"], answers) if a and a.strip()
            ]
        st.session_state["qa_pairs"] = qa_pairs

        try:
            llm = load_llm()
            with st.spinner("Analyzing trade-offs, risks, and uncertainty..."):
                analysis = run_analysis_stage(llm, context, st.session_state["qa_pairs"])
                st.session_state["tradeoffs"] = analysis["tradeoffs"]
                st.session_state["risks"] = analysis["risks"]
                st.session_state["uncertainties"] = analysis["uncertainties"]

            with st.spinner("Writing the final decision report..."):
                final_report = run_final_report_stage(
                    llm, context, analysis, st.session_state["missing_info"], st.session_state["qa_pairs"]
                )
                st.session_state["final_report"] = final_report

            st.session_state["stage"] = "report"
            st.rerun()

        except Exception as e:
            st.error(f"Something went wrong while analyzing your decision: {e}")


# ---------------------------------------------------------------------------
# Stage: REPORT
# ---------------------------------------------------------------------------
elif st.session_state["stage"] == "report":
    context = {
        "decision": st.session_state["understanding"]["decision"],
        "category": st.session_state["understanding"]["category"],
        "objective": st.session_state["understanding"]["objective"],
        "options": st.session_state["options"],
        "criteria": st.session_state["criteria"],
        "constraints": st.session_state["constraints"],
    }
    render_decision_canvas(context)

    # If we jumped here directly from "input" (sufficient info, no questions asked yet),
    # run the analysis + final report now.
    if st.session_state["final_report"] is None:
        try:
            llm = load_llm()
            with st.spinner("Analyzing trade-offs, risks, and uncertainty..."):
                analysis = run_analysis_stage(llm, context, st.session_state["qa_pairs"])
                st.session_state["tradeoffs"] = analysis["tradeoffs"]
                st.session_state["risks"] = analysis["risks"]
                st.session_state["uncertainties"] = analysis["uncertainties"]

            with st.spinner("Writing the final decision report..."):
                final_report = run_final_report_stage(
                    llm, context, analysis, st.session_state["missing_info"], st.session_state["qa_pairs"]
                )
                st.session_state["final_report"] = final_report
        except Exception as e:
            st.error(f"Something went wrong while generating the report: {e}")
            st.stop()

    st.markdown("### 📊 Analysis")

    tab_tradeoffs, tab_risks, tab_uncertainty, tab_missing = st.tabs(
        ["💰 Trade-offs", "⚠️ Risks", "❓ Uncertainty", "🧩 Missing Information"]
    )

    with tab_tradeoffs:
        for t in st.session_state["tradeoffs"]:
            gains_html = bullet_list(t["gains"])
            sacrifices_html = bullet_list(t["sacrifices"])
            st.markdown(
                f"""<div class="df-card df-accent-blue">
                <h4>{t['option']}</h4>
                <b>Gains</b><ul>{gains_html}</ul>
                <b>Sacrifices</b><ul>{sacrifices_html}</ul>
                </div>""",
                unsafe_allow_html=True,
            )

    with tab_risks:
        for r in st.session_state["risks"]:
            risks_html = bullet_list(r["risks"])
            st.markdown(
                f"""<div class="df-card df-accent-red"><h4>{r['option']}</h4><ul>{risks_html}</ul></div>""",
                unsafe_allow_html=True,
            )

    with tab_uncertainty:
        uncertainty_html = bullet_list(st.session_state["uncertainties"])
        st.markdown(f"""<div class="df-card df-accent-purple"><ul>{uncertainty_html}</ul></div>""", unsafe_allow_html=True)

    with tab_missing:
        report = st.session_state["final_report"]
        missing_html = bullet_list(report["missing_information"], empty_text="None - all key information was available.")
        st.markdown(f"""<div class="df-card df-accent-amber"><ul>{missing_html}</ul></div>""", unsafe_allow_html=True)

    st.markdown("### 📝 Final Decision Report")
    report = st.session_state["final_report"]

    st.markdown("#### Comparison")
    st.write(report["comparison"])

    st.markdown("#### Questions to Consider")
    for q in report["questions_to_consider"]:
        st.markdown(f"- {q}")

    st.markdown("#### Neutral Summary")
    st.markdown(f'<div class="df-callout">{report["neutral_summary"]}</div>', unsafe_allow_html=True)

    if st.session_state["qa_pairs"]:
        with st.expander("💬 Conversation History (follow-up questions & answers)"):
            for q, a in st.session_state["qa_pairs"]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")

    st.divider()
    if st.button("🔄 Start New Decision", type="primary"):
        start_new_decision()
        st.rerun()