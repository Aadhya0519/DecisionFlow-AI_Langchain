"""
Pydantic models used as structured outputs for every LangChain chain in
DecisionFlow AI.

Each chain is bound to exactly one of these schemas via
`llm.with_structured_output(SchemaClass)`, so the LLM is forced to return
data in this shape instead of free-form text.
"""

from typing import List
from pydantic import BaseModel, Field


class DecisionUnderstanding(BaseModel):
    """Structured understanding of the raw decision text entered by the user."""

    decision: str = Field(
        description="A clear, one-sentence restatement of the decision being made."
    )
    category: str = Field(
        description=(
            "Exactly one category: Career, Education, Finance, Technology, "
            "Purchase, Business, Lifestyle, or Other."
        )
    )
    objective: str = Field(
        description="What the user is fundamentally trying to achieve with this decision."
    )


class OptionList(BaseModel):
    """The distinct options/alternatives identified in the decision."""

    options: List[str] = Field(
        description="2 to 5 short, clearly worded options the user is choosing between."
    )


class CriteriaList(BaseModel):
    """The factors relevant to comparing the options."""

    criteria: List[str] = Field(
        description=(
            "3 to 6 decision-specific criteria (e.g. cost, time, growth potential, "
            "risk, convenience). Must be specific to this decision, not generic filler."
        )
    )


class ConstraintList(BaseModel):
    """Constraints that limit the decision, if any."""

    constraints: List[str] = Field(
        description=(
            "Budget, time, skill, deadline, or resource constraints that are "
            "explicitly or implicitly mentioned. Empty list if none are present."
        )
    )


class SufficiencyCheck(BaseModel):
    """Whether enough information exists to run a meaningful analysis."""

    is_sufficient: bool = Field(
        description=(
            "True if there is enough information to produce a meaningful trade-off, "
            "risk, and uncertainty analysis. False if key missing information would "
            "materially change the analysis."
        )
    )
    missing_info: List[str] = Field(
        description="Specific pieces of missing information that matter for this decision. Empty list if sufficient."
    )
    reasoning: str = Field(
        description="One or two sentences explaining the sufficiency judgment."
    )


class FollowUpQuestions(BaseModel):
    """Dynamically generated follow-up questions tailored to the specific decision."""

    questions: List[str] = Field(
        description=(
            "3 to 6 short, specific questions tailored exactly to this decision "
            "and its missing information. Empty list if none are needed."
        )
    )


class OptionTradeoff(BaseModel):
    """What is gained and sacrificed for a single option."""

    option: str = Field(description="The option name, matching one of the identified options exactly.")
    gains: List[str] = Field(description="What is gained by choosing this option.")
    sacrifices: List[str] = Field(description="What is given up or sacrificed by choosing this option.")


class TradeoffAnalysis(BaseModel):
    tradeoffs: List[OptionTradeoff] = Field(
        description="One trade-off entry per identified option."
    )


class OptionRisk(BaseModel):
    """Risks specific to a single option."""

    option: str = Field(description="The option name, matching one of the identified options exactly.")
    risks: List[str] = Field(description="Potential risks specific to choosing this option.")


class RiskAnalysis(BaseModel):
    risk_analysis: List[OptionRisk] = Field(
        description="One risk entry per identified option."
    )


class UncertaintyAnalysis(BaseModel):
    uncertainties: List[str] = Field(
        description=(
            "Things that genuinely cannot be determined from the information "
            "provided, and that would affect the decision."
        )
    )


class FinalReport(BaseModel):
    """The complete, structured decision report shown to the user."""

    decision: str
    category: str
    objective: str
    options: List[str]
    criteria: List[str]
    constraints: List[str]
    tradeoffs: List[OptionTradeoff]
    risks: List[OptionRisk]
    uncertainties: List[str]
    missing_information: List[str] = Field(
        description="Important information that is still unavailable, even after any follow-up answers."
    )
    comparison: str = Field(
        description="A clear, neutral side-by-side comparison paragraph of the options."
    )
    questions_to_consider: List[str] = Field(
        description="Reflective questions the user should think about before deciding."
    )
    neutral_summary: str = Field(
        description=(
            "A neutral summary of the whole analysis. Must NOT declare a winner "
            "or make the decision for the user."
        )
    )
