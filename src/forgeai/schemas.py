"""Core Pydantic schemas for ForgeAI's declarative pipeline artifacts."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class DocProfile(BaseModel):
    """Lightweight profile of a single document in the knowledge folder."""

    filename: str
    file_type: Literal["pdf", "csv", "txt", "md"]
    title: str
    headings: list[str] = Field(default_factory=list)
    summary_bullets: list[str] = Field(default_factory=list)
    word_count: int = 0
    # CSV-specific metadata
    csv_columns: list[str] = Field(default_factory=list)
    csv_row_count: int = 0
    csv_sample_rows: list[dict[str, str]] = Field(default_factory=list)


class CorpusManifest(BaseModel):
    """Summary manifest of the entire document folder."""

    docs_dir: str
    total_files: int
    documents: list[DocProfile] = Field(default_factory=list)


class ClarifyingQA(BaseModel):
    """A single clarifying question and answer from the intake interview."""

    question: str
    answer: str


class UnsupportedGap(BaseModel):
    """A brief requirement that cannot be grounded in the provided documents."""

    requested_topic: str
    reason: str


class RequirementsDoc(BaseModel):
    """Structured requirements produced by the Intake Interviewer & Groundability Guard."""

    brief: str
    agent_name: str
    scope: str
    audience: str
    tone: str
    must_answer_topics: list[str] = Field(default_factory=list)
    must_refuse_topics: list[str] = Field(default_factory=list)
    tool_supported_capabilities: list[str] = Field(default_factory=list)
    unsupported_gaps: list[UnsupportedGap] = Field(default_factory=list)
    clarifying_qa: list[ClarifyingQA] = Field(default_factory=list, max_length=5)


class RetrievalConfig(BaseModel):
    """Chunking and retrieval parameters for the generated agent."""

    chunk_size: int = Field(default=500, ge=150, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=400)
    top_k: int = Field(default=4, ge=1, le=15)
    retrieval_mode: Literal["hybrid", "bm25", "tfidf"] = "hybrid"


class SystemPromptSections(BaseModel):
    """Modular system prompt sections enabling surgical critic-optimizer patches."""

    role: str
    instructions: str
    guardrails: str
    refusal_policy: str
    few_shot_examples: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Compile modular sections into a single system prompt string."""
        parts = [
            f"## Role\n{self.role}",
            f"## Instructions\n{self.instructions}",
            f"## Grounding & Guardrails\n{self.guardrails}",
            f"## Refusal Policy\n{self.refusal_policy}",
        ]
        if self.few_shot_examples:
            examples_block = "\n\n".join(self.few_shot_examples)
            parts.append(f"## Examples\n{examples_block}")
        return "\n\n".join(parts)


class ToolSelection(BaseModel):
    """A tool selected from the registry with explicit justification from the brief."""

    name: Literal["csv_lookup", "calculator", "open_meteo_weather", "current_date"]
    justification: str


class OutputContract(BaseModel):
    """Expected output format and citation rules for the generated agent."""

    format: Literal["markdown", "plain_text", "json"] = "markdown"
    require_citations: bool = True
    citation_format: str = "[Source: <filename>]"
    additional_rules: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Complete declarative agent definition produced by ForgeAI."""

    agent_name: str
    version: int = 1
    description: str
    system_prompt: SystemPromptSections
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    selected_tools: list[ToolSelection] = Field(default_factory=list)
    output_contract: OutputContract = Field(default_factory=OutputContract)


class DocumentChunk(BaseModel):
    """An indexed chunk of text from a source document."""

    chunk_id: str
    filename: str
    page_or_row: str
    text: str


class EvalCase(BaseModel):
    """A single auto-generated evaluation test case."""

    id: str
    category: Literal["answerable", "refusal", "edge_multi_doc", "edge_ambiguous"]
    question: str
    expected_source_docs: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    source_chunk_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class EvalSet(BaseModel):
    """Collection of auto-generated evaluation cases."""

    agent_name: str
    cases: list[EvalCase] = Field(default_factory=list)


class RequirementCoverage(BaseModel):
    """Inspectable link from a brief requirement to eval evidence."""

    topic: str
    requirement_type: Literal["must_answer", "must_refuse", "tool_capability", "unsupported_gap"]
    status: Literal["covered", "untested"]
    matching_case_ids: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)


class CoverageMatrix(BaseModel):
    """Heuristic traceability audit; coverage means a related eval exists, not that it passed."""

    matching_method: str = "normalized keyword overlap; inspect matches before treating them as semantic proof"
    requirements: list[RequirementCoverage] = Field(default_factory=list)


class FactValidationResult(BaseModel):
    """Support check of a single expected_fact against its cited chunk text."""

    fact: str
    supported: bool
    answers_question: bool = True
    evidence_quote: str = ""


class EvalCaseValidation(BaseModel):
    """Validation audit record for a single non-refusal eval case."""

    case_id: str
    chunk_ids_valid: bool
    original_fact_count: int
    kept_fact_count: int
    dropped_facts: list[str] = Field(default_factory=list)
    regenerated: bool = False


class EvalValidationReport(BaseModel):
    """Summary of chunk-grounding validation across the generated eval set."""

    total_non_refusal_cases: int
    cases_validated: int
    total_facts_checked: int
    facts_kept: int
    facts_dropped: int
    cases_regenerated: int
    case_details: list[EvalCaseValidation] = Field(default_factory=list)


class ToolCallTrace(BaseModel):
    """Record of a tool invocation during an agent run."""

    tool_name: str
    arguments: dict
    result: str


class AgentTrace(BaseModel):
    """Execution trace of the generic agent runner for a single query."""

    question: str
    answer: str
    retrieved_chunks: list[DocumentChunk] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: int = 0


class FactCheck(BaseModel):
    """Binary verification of a single atomic expected fact."""

    fact: str
    present: bool
    evidence: str = ""


FailureCategory = Literal[
    "retrieval_miss",
    "hallucination",
    "wrong_refusal",
    "prompt_defect",
]


class JudgeResult(BaseModel):
    """Hybrid deterministic + decomposed LLM judge output for a single eval case."""

    case_id: str
    category: str
    passed: bool
    retrieval_hit: bool
    citation_present: bool
    refused: bool
    refusal_correct: bool
    fact_checks: list[FactCheck] = Field(default_factory=list)
    fact_score: float = 0.0
    grounded: bool = True
    failure_category: Optional[FailureCategory] = None
    failure_reason: str = ""


class IterationDiff(BaseModel):
    """A targeted mutation applied by the critic-optimizer."""

    component: str
    before: str
    after: str
    reason: str


class IterationLog(BaseModel):
    """Audit log for one critic-optimizer iteration."""

    iteration: int
    score_before: float
    score_after: float
    passed_before: int
    passed_after: int
    total_cases: int
    failure_breakdown: dict[str, int] = Field(default_factory=dict)
    addressed_case_ids: list[str] = Field(default_factory=list)
    changes: list[IterationDiff] = Field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0.0


class RunBudget(BaseModel):
    """Token and latency budget tracking for early stopping."""

    max_tokens: int = 250000
    max_latency_seconds: float = 900.0
    tokens_used: int = 0
    latency_seconds: float = 0.0
    stopped_early: bool = False
    stop_reason: str = ""


class Scorecard(BaseModel):
    """Final evaluation summary across all iterations."""

    agent_name: str
    total_cases: int
    initial_pass_rate: float
    final_pass_rate: float
    passed_cases: int
    failed_cases: int
    iterations: list[IterationLog] = Field(default_factory=list)
    final_results: list[JudgeResult] = Field(default_factory=list)
    budget: RunBudget = Field(default_factory=RunBudget)
