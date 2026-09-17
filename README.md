# DecisionFlow AI — Intelligent Decision Analysis Engine

*"Turn complex choices into structured decisions."*

DecisionFlow AI takes any real-world decision described in plain English —
*"Should I buy a new laptop or take a certification?"* — and turns it into a
structured decision report: options, criteria, constraints, trade-offs,
risks, uncertainties, and a neutral side-by-side comparison. It never tells
you what to choose — it structures the choice for you.

This is a **LangChain-only** project built to demonstrate orchestration:
LCEL, sequential chains, parallel chains, branching, structured (Pydantic)
outputs, and conversational state — with **no RAG, no vector databases, and
no autonomous agents**.

---

## 1. Project structure

```text
DecisionFlow_AI/
│
├── app.py                      # Streamlit UI and app orchestration
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── chains/
│   ├── decision_chain.py       # Decision Understanding Chain
│   ├── option_chain.py         # Option Extraction Chain
│   ├── criteria_chain.py       # Criteria Identification Chain
│   ├── constraint_chain.py     # Constraint Extraction Chain
│   ├── sufficiency_chain.py    # Information Sufficiency Chain
│   ├── question_chain.py       # Follow-up Question Chain
│   ├── tradeoff_chain.py       # Trade-off Analysis Chain
│   ├── risk_chain.py           # Risk Analysis Chain
│   ├── uncertainty_chain.py    # Uncertainty Analysis Chain
│   ├── final_report_chain.py   # Final Decision Report Chain
│   └── pipeline.py             # Wires all chains together (RunnableParallel / RunnableBranch)
│
├── models/
│   └── schemas.py              # Pydantic models for every structured output
│
├── prompts/
│   └── prompts.py              # ChatPromptTemplate for every chain
│
├── utils/
│   ├── llm.py                  # LLM provider setup (isolated, swappable via .env)
│   └── session.py               # Streamlit session-state helpers
│
└── config/
    └── settings.py             # App name, tagline, categories, example decisions
```

---

## 2. Installation

```bash
# 1. Clone / open the project folder
cd DecisionFlow_AI

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# then open .env and set LLM_API_KEY to your OpenAI (or OpenAI-compatible) API key
```

## 3. Running the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 4. Example decisions to try

- "Should I buy a new laptop or take a certification?"
- "Should I learn Data Science or Data Engineering?"
- "Should I rent a house closer to work or stay farther away?"
- "Should I launch my project now or wait?"
- "Should I use PostgreSQL or MySQL for my new project?"
- "Should I spend my savings on a course or keep the money?"

---

## 5. How the LangChain pipeline works

```text
User Decision
      │
      ▼
Understanding Stage  ── RunnableParallel ──▶  Decision Understanding Chain
      │                                       Option Extraction Chain
      │                                       Criteria Identification Chain
      │                                       Constraint Extraction Chain
      ▼
Information Sufficiency Chain
      │
      ▼
RunnableBranch ──▶  insufficient? ──▶ Follow-up Question Chain ──▶ (user answers)
      │
      ▼ sufficient (or after answers)
Analysis Stage  ── RunnableParallel ──▶  Trade-off Analysis Chain
      │                                  Risk Analysis Chain
      │                                  Uncertainty Analysis Chain
      ▼
Final Decision Report Chain
      │
      ▼
Structured Decision Report (rendered in Streamlit)
```

**Sequential chains** — each chain in `chains/` is built with LCEL's pipe
syntax: `prompt | llm.with_structured_output(Schema)`. The prompt template
formats the inputs, and the model is forced to return data matching a
Pydantic schema instead of free text.

**Parallel chains** — `chains/pipeline.py` uses `RunnableParallel` twice:
once to run the Decision Understanding, Option Extraction, Criteria
Identification, and Constraint Extraction chains together in the
Understanding Stage (they're independent of each other), and once to run
the Trade-off, Risk, and Uncertainty chains together in the Analysis Stage.

**Branching** — `chains/pipeline.py` uses `RunnableBranch` to route between
the Follow-up Question Chain and an empty result, based on the output of
the Information Sufficiency Chain. If information is missing, the branch
calls the question chain; otherwise, no questions are generated and the app
skips straight to analysis.

**Structured outputs** — every chain's final step is
`llm.with_structured_output(SomePydanticModel)`, defined in
`models/schemas.py`. This is used throughout instead of parsing free text.

**Conversation / session state** — `utils/session.py` manages the
Streamlit `session_state` so a decision session moves through three stages
(`input` → `questions` → `report`), preserves follow-up Q&A as
conversational context fed into later chains, and keeps a history of past
decisions for the current session.

**Streamlit UI** — `app.py` is a polished, SaaS-style interface with a
Decision Canvas, tabbed Analysis section (trade-offs / risks / uncertainty /
missing information), a Final Decision Report, and a session history
sidebar.

---

## 6. Notes

- Swap LLM providers by editing only `utils/llm.py` and your `.env` file —
  no chain code needs to change. Any OpenAI-compatible endpoint (OpenAI,
  Groq, Together AI, OpenRouter, etc.) works by setting `LLM_BASE_URL`.
- This project intentionally does **not** use RAG, vector stores, embeddings,
  web search, or autonomous agents — the goal is to demonstrate LangChain
  chain orchestration itself.
