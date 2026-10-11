"""
Pydantic Schemas for the BODMAS Multi-Stage Pipeline.
These schemas are used for strict structured JSON outputs with Gemma 4 via Gemini API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# --- Stage 1: Parser & Validator Schema ---
class ValidationOutput(BaseModel):
    is_valid: bool = Field(
        ...,
        description="True if the input is a mathematically valid arithmetic expression using BODMAS, False otherwise."
    )
    clean_equation: str = Field(
        ...,
        description="The cleaned and standardized equation string (e.g. spaces normalized, valid operators)."
    )
    error_message: Optional[str] = Field(
        None,
        description="Human-readable reason if the equation is invalid (e.g., mismatched brackets, trailing operator, division by zero)."
    )


# --- Stage 2: Step-by-Step Algebraic Engine Schema ---
class BODMASStep(BaseModel):
    step_index: int = Field(..., description="1-indexed step number.")
    rule: str = Field(
        ...,
        description="The specific BODMAS rule applied in this step: 'Brackets', 'Orders', 'Division', 'Multiplication', 'Addition', or 'Subtraction'."
    )
    focus: str = Field(
        ...,
        description="The exact sub-expression evaluated in this step (e.g. '2 - 8' or '5 * -6')."
    )
    result: str = Field(
        ...,
        description="The numerical outcome of evaluating the focus sub-expression (e.g. '-6' or '-30')."
    )
    current_equation: str = Field(
        ...,
        description="The full equation rewritten after substituting the focus with its result."
    )


class ReductionOutput(BaseModel):
    original_equation: str = Field(..., description="The original cleaned equation.")
    steps: List[BODMASStep] = Field(..., description="Ordered list of reduction steps until fully solved.")
    final_answer: str = Field(..., description="The final single number result as a string.")


# --- Stage 3: Friendly Tutor Explainer Schema ---
class TutorStepExplanation(BaseModel):
    step_index: int = Field(..., description="Corresponding step number.")
    heading: str = Field(..., description="Short catchy step title, e.g., 'Step 1: Simplify inside Brackets'.")
    explanation: str = Field(..., description="Clear, student-friendly explanation of why this step was taken according to BODMAS.")
    tip: str = Field(..., description="A helpful mental math shortcut or warning to avoid common student pitfalls.")


class TutorOutput(BaseModel):
    title: str = Field(..., description="Catchy title for the walkthrough.")
    overview: Optional[str] = Field(None, description="High-level summary of the equation and the approach taken.")
    explanation: Optional[str] = Field(None, description="Clear, student-friendly explanation of how the equation is resolved according to BODMAS.")
    step_explanations: Optional[List[TutorStepExplanation]] = Field(default_factory=list, description="Pedagogical explanations for each step.")
    tip: Optional[str] = Field(None, description="A helpful mental math shortcut or warning to avoid common student pitfalls.")
    encouragement: Optional[str] = Field(None, description="Positive closing words or key takeaway for the student.")


# --- Full Pipeline Output Contract ---
class TelemetryData(BaseModel):
    total_latency_ms: float
    stage1_latency_ms: float
    stage2_latency_ms: float
    stage3_latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_used: str
    schema_valid: bool


class PipelineResult(BaseModel):
    status: str  # "success" or "invalid_input" or "error"
    clean_equation: Optional[str] = None
    final_answer: Optional[str] = None
    reduction_steps: Optional[List[BODMASStep]] = None
    tutor_explanation: Optional[TutorOutput] = None
    error: Optional[str] = None
    telemetry: TelemetryData
