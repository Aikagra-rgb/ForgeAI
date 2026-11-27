"""Intake Interviewer & Tool-Aware Groundability Guard."""

from __future__ import annotations

import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

from forgeai.llm import LLMClient
from forgeai.schemas import ClarifyingQA, CorpusManifest, RequirementsDoc, UnsupportedGap
from forgeai.tools import TOOL_DEFINITIONS, format_tool_catalog


class IntakeProbe(BaseModel):
    clarifying_questions: list[str] = Field(default_factory=list, max_length=5)
    tool_supported_capabilities: list[str] = Field(default_factory=list)
    unsupported_gaps: list[UnsupportedGap] = Field(default_factory=list)


_IntakeProbe = IntakeProbe


_REFUSAL_CLAUSE_RE = re.compile(
    r"\b(?:must\s+refuse|should\s+refuse|do\s+not\s+answer|refuse\s+requests|cannot\s+answer|not\s+covered)\b",
    re.IGNORECASE,
)


def _split_affirmative_clauses(brief: str) -> str:
    """Return sentences/clauses from brief that are affirmative requests, excluding refusal clauses."""
    clauses = re.split(r"[.;\n]+", brief)
    affirmative = [c.strip() for c in clauses if c.strip() and not _REFUSAL_CLAUSE_RE.search(c)]
    return " ".join(affirmative).lower()


_EXTERNAL_ACTION_HINTS = (
    "jira",
    "ticket",
    "booking",
    "reserv",
    "shipping",
    "tax",
    "salary",
    "crew contract",
    "email",
    "submit",
    "send",
    "create report",
    "scada",
    "circuit breaker",
    "dispatch",
    "ecrf",
    "medidata",
    "concur",
    "irb",
    "manifest",
)

_EXTERNAL_WRITE_ACTION_PATTERNS = (
    re.compile(r"\b(?:remotely\s+trip|trip\s+circuit\s+breaker|dispatch\s+(?:scada|inverter|setpoint)|scada\b)", re.I),
    re.compile(r"\b(?:book|reserve)\s+(?:a\s+|my\s+)?(?:flight|flights|berth|berths|hotel|hotels|ticket|tickets|reservation|slot)\b", re.I),
    re.compile(r"\b(?:submit|file|approve)\s+(?:[a-z0-9_/-]+\s+){0,4}(?:ticket|tickets|report|reports|manifest|manifests|order|orders|claim|claims|approval)\b", re.I),
    re.compile(r"\b(?:update|modify|delete|write)\s+(?:[a-z0-9_/-]+\s+){0,4}(?:record|records|ecrf|database|crm|erp|jira|concur|medidata)\b", re.I),
    re.compile(r"\b(?:send|draft\s+and\s+send)\s+(?:an?\s+)?(?:email|emails|slack|sms|message)\b", re.I),
    re.compile(r"\b(?:issue|sign)\s+(?:[a-z0-9_/-]+\s+){0,4}(?:approval\s+letter|certificate|purchase\s+order|invoice)\b", re.I),
)


def _detect_external_write_gaps(brief: str) -> list[UnsupportedGap]:
    """Deterministically detect affirmative external write/control clauses a small model may miss."""
    gaps: list[UnsupportedGap] = []
    for raw_clause in re.split(r"[.;\n]+", brief):
        clause = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", raw_clause).strip()
        if not clause or _REFUSAL_CLAUSE_RE.search(clause):
            continue
        if any(pat.search(clause) for pat in _EXTERNAL_WRITE_ACTION_PATTERNS):
            gaps.append(
                UnsupportedGap(
                    requested_topic=clause,
                    reason=(
                        "Requires an external write, booking, or control-system action not supported "
                        "by the provided documents or the 4 read-only registered tools."
                    ),
                )
            )
    return gaps


def _merge_missing_external_gaps(brief: str, existing: list[UnsupportedGap]) -> list[UnsupportedGap]:
    """Append deterministically detected external write gaps not already covered."""
    merged = list(existing)
    for cand in _detect_external_write_gaps(brief):
        cand_tokens = _content_tokens(cand.requested_topic)
        already = any(
            len(cand_tokens & _content_tokens(g.requested_topic)) >= 2
            or cand.requested_topic.lower() in g.requested_topic.lower()
            or g.requested_topic.lower() in cand.requested_topic.lower()
            for g in merged
        )
        if not already:
            merged.append(cand)
    return merged

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "based",
    "data", "file", "files", "info", "information", "requests", "request",
    "handle", "using", "via", "tool", "tools", "documents", "unsupported",
}


def _content_tokens(text: str) -> set[str]:
    """Extract normalized content tokens (simple singular/stem normalization) for overlap checks."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    out: set[str] = set()
    for tok in raw:
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("es") and len(tok) > 4:
            tok = tok[:-2]
        elif tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss"):
            tok = tok[:-1]
        out.add(tok)
    return out


def _is_external_action(text: str) -> bool:
    low = text.lower()
    return any(ext in low for ext in _EXTERNAL_ACTION_HINTS)


def _reconcile_tool_capabilities(
    brief: str,
    manifest: CorpusManifest,
    req: RequirementsDoc,
) -> RequirementsDoc:
    """Reconcile tool-supported capabilities, false CSV gaps, and contradictory must_answer topics."""
    affirmative = _split_affirmative_clauses(brief)
    has_csv = any(d.file_type == "csv" for d in manifest.documents)
    tool_names = tuple(TOOL_DEFINITIONS.keys())

    # 1. Retain only tool_supported_capabilities that actually reference a registered tool name
    req.tool_supported_capabilities = [
        cap for cap in req.tool_supported_capabilities
        if any(tname in cap.lower() for tname in tool_names)
    ]

    tool_patterns: dict[str, tuple[tuple[str, ...], str]] = {
        "open_meteo_weather": (
            (r"\bweather\b", r"\bforecast\b", r"\btemperature\b", r"\bwind\b", r"\bprecipitation\b", r"\bmeteorolog"),
            "Live weather lookup via open_meteo_weather tool",
        ),
        "calculator": (
            (r"\bcalculat", r"\bcomput", r"\bmath\b", r"\barithmetic\b", r"\btotal\b", r"\bsurcharge\b", r"\bdiscount\b", r"\bcost estimate"),
            "Numeric calculation via calculator tool",
        ),
        "csv_lookup": (
            (r"\bcsv\b", r"\blook\s*up\b", r"\blookup\b", r"\bcatalog(?:ue)?\b", r"\binventory\b", r"\bdataset\b", r"\bspreadsheet\b", r"\btabular\b"),
            "Structured tabular lookup via csv_lookup tool",
        ),
        "current_date": (
            (r"\btoday\b", r"\bcurrent date\b", r"\bas of today\b", r"\bexpir", r"\bcurrent time\b"),
            "Current date/time awareness via current_date tool",
        ),
    }

    active_patterns: list[str] = []
    existing_caps = {c.lower() for c in req.tool_supported_capabilities}

    for tool_name, (patterns, label) in tool_patterns.items():
        if tool_name not in TOOL_DEFINITIONS:
            continue
        if tool_name == "csv_lookup" and not has_csv:
            continue
        if any(re.search(pat, affirmative) for pat in patterns):
            active_patterns.extend(patterns)
            if not any(re.search(pat, cap) for cap in existing_caps for pat in patterns) and label.lower() not in existing_caps:
                req.tool_supported_capabilities.append(label)
                existing_caps.add(label.lower())

    # 2. Remove false unsupported_gaps / must_refuse_topics for affirmatively requested registered tools
    if active_patterns:
        def _matches_active_tool(text: str) -> bool:
            if _is_external_action(text):
                return False
            return any(re.search(pat, text.lower()) for pat in active_patterns)

        req.unsupported_gaps = [
            gap for gap in req.unsupported_gaps
            if not _matches_active_tool(gap.requested_topic)
        ]
        req.must_refuse_topics = [
            topic for topic in req.must_refuse_topics
            if not _matches_active_tool(topic)
        ]

    # 3. Drop false unsupported_gaps when CSV column headers/filenames directly cover the topic
    csv_tokens: set[str] = set()
    for doc in manifest.documents:
        if doc.file_type == "csv":
            csv_tokens.update(_content_tokens(doc.filename.rsplit(".", 1)[0].replace("_", " ")))
            for col in doc.csv_columns:
                csv_tokens.update(_content_tokens(col.replace("_", " ")))

    if csv_tokens:
        dropped_gap_topics: list[str] = []
        kept_gaps: list[UnsupportedGap] = []
        for gap in req.unsupported_gaps:
            if _is_external_action(gap.requested_topic):
                kept_gaps.append(gap)
                continue
            topic_tokens = _content_tokens(gap.requested_topic)
            if len(topic_tokens & csv_tokens) >= 2:
                dropped_gap_topics.append(gap.requested_topic)
            else:
                kept_gaps.append(gap)
        req.unsupported_gaps = kept_gaps

        if dropped_gap_topics:
            dropped_token_sets = [_content_tokens(t) for t in dropped_gap_topics]
            req.must_refuse_topics = [
                topic for topic in req.must_refuse_topics
                if not any(len(_content_tokens(topic) & dt) >= 2 for dt in dropped_token_sets)
            ]

    # 3b. Deterministic safety net: ensure affirmative external write/control clauses are flagged even on small 4B models
    req.unsupported_gaps = _merge_missing_external_gaps(brief, req.unsupported_gaps)

    # 4. Eliminate contradictory must_answer_topics that overlap with unsupported_gaps or must_refuse_topics
    refusal_sources = [g.requested_topic for g in req.unsupported_gaps] + list(req.must_refuse_topics)
    refusal_token_sets = [_content_tokens(t) for t in refusal_sources if _content_tokens(t)]
    if refusal_token_sets:
        cleaned_must_answer: list[str] = []
        for topic in req.must_answer_topics:
            t_tokens = _content_tokens(topic)
            overlaps = any(
                len(t_tokens & r_tokens) >= 2
                or (len(t_tokens & r_tokens) == 1 and any(ext in topic.lower() for ext in _EXTERNAL_ACTION_HINTS))
                for r_tokens in refusal_token_sets
            )
            if not overlaps:
                cleaned_must_answer.append(topic)
        req.must_answer_topics = cleaned_must_answer

    return req


def probe_intake(
    brief: str,
    manifest: CorpusManifest,
    llm: LLMClient,
) -> IntakeProbe:
    """Step 1 of intake: inspect brief + manifest + registered tools to propose <=5 questions and gaps."""
    manifest_json = manifest.model_dump_json(indent=2)
    tool_catalog = format_tool_catalog()

    probe_messages = [
        {
            "role": "system",
            "content": (
                "You are ForgeAI's Intake Interviewer & Tool-Aware Groundability Guard.\n"
                "Compare the user's brief against BOTH the Corpus Manifest AND Available Registered Tools.\n"
                "1. Before placing ANY topic in `unsupported_gaps`, carefully inspect EVERY file's title, headings, "
                "summary bullets, AND CSV column names/sample rows in `Corpus Manifest` (semantic equivalents count: "
                "e.g., `Vessel class` + `Capacity FFE` in `fleet_data.csv` directly support container fleet vessel classes/capacities).\n"
                "2. A requested capability is SUPPORTED if it can be fulfilled by the documents in the "
                "Corpus Manifest OR by one of the 4 Available Registered Tools (`csv_lookup`, `calculator`, "
                "`open_meteo_weather`, `current_date`). List ONLY capabilities explicitly fulfilled by those 4 registered "
                "tools in `tool_supported_capabilities`, naming the exact tool (e.g., '... via calculator'). "
                "Leave `tool_supported_capabilities` empty `[]` if no tools are needed.\n"
                "3. List a requested topic in `unsupported_gaps` if NEITHER the documents NOR those 4 registered "
                "tools can perform it. CRITICAL: External write actions or third-party integrations NOT in the 4 "
                "registered tools (e.g., submitting Jira tickets, sending emails, booking reservations, placing orders, "
                "live shipping APIs) MUST be placed in `unsupported_gaps`—NEVER in `tool_supported_capabilities`, even "
                "if a policy document mentions them.\n"
                "4. Propose at most 5 concise clarifying questions (about audience, tone, refusal boundaries, "
                "or edge behavior) in `clarifying_questions`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brief:\n{brief}\n\n"
                f"Corpus Manifest:\n{manifest_json}\n\n"
                f"Available Registered Tools:\n{tool_catalog}"
            ),
        },
    ]

    probe = llm.structured_chat(probe_messages, IntakeProbe, temperature=0.0)
    probe.clarifying_questions = probe.clarifying_questions[:5]
    probe.unsupported_gaps = _merge_missing_external_gaps(brief, probe.unsupported_gaps)
    return probe


def synthesize_requirements(
    brief: str,
    manifest: CorpusManifest,
    probe: IntakeProbe,
    qa_pairs: list[ClarifyingQA],
    llm: LLMClient,
) -> RequirementsDoc:
    """Step 2 of intake: synthesize and reconcile a structured RequirementsDoc from probe + Q&A."""
    manifest_json = manifest.model_dump_json(indent=2)
    tool_catalog = format_tool_catalog()
    qa_text = "\n".join(f"Q: {x.question}\nA: {x.answer}" for x in qa_pairs) or "(No clarifying Q&A)"
    gaps_text = (
        "\n".join(f"- {g.requested_topic}: {g.reason}" for g in probe.unsupported_gaps)
        or "(None detected)"
    )
    tool_caps_text = (
        "\n".join(f"- {c}" for c in probe.tool_supported_capabilities)
        or "(None identified)"
    )

    synth_messages = [
        {
            "role": "system",
            "content": (
                "You are ForgeAI's Requirements Architect.\n"
                "Synthesize a crisp, structured RequirementsDoc from the brief, corpus manifest, "
                "registered tools, tool-supported capabilities, groundability gaps, and clarifying Q&A.\n"
                "CRITICAL RULES:\n"
                "1. Preserve `tool_supported_capabilities` ONLY for capabilities fulfilled by the 4 registered tools.\n"
                "2. Never put any topic from `unsupported_gaps` or `must_refuse_topics` into `must_answer_topics`.\n"
                "3. Drop any false gap from `unsupported_gaps` if a document's CSV columns or text clearly cover that concept.\n"
                "4. Do NOT turn unanswered clarifying questions or tool-supported capabilities into `must_refuse_topics`, "
                "and NEVER reference 'clarifying questions' or 'conservative defaults' in policy/scope text.\n"
                "5. Only place genuinely unsupported gaps and topics explicitly excluded by the brief into `must_refuse_topics`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brief:\n{brief}\n\n"
                f"Corpus Manifest:\n{manifest_json}\n\n"
                f"Available Registered Tools:\n{tool_catalog}\n\n"
                f"Tool-Supported Capabilities:\n{tool_caps_text}\n\n"
                f"Detected Unsupported Gaps:\n{gaps_text}\n\n"
                f"Clarifying Q&A:\n{qa_text}"
            ),
        },
    ]

    req = llm.structured_chat(synth_messages, RequirementsDoc, temperature=0.0)
    # Deterministic safety nets
    req.brief = brief
    req.clarifying_qa = qa_pairs[:5]
    if probe.tool_supported_capabilities and not req.tool_supported_capabilities:
        req.tool_supported_capabilities = probe.tool_supported_capabilities
    if probe.unsupported_gaps and not req.unsupported_gaps:
        req.unsupported_gaps = probe.unsupported_gaps

    req = _reconcile_tool_capabilities(brief, manifest, req)

    existing_refuse = {t.lower() for t in req.must_refuse_topics}
    for gap in req.unsupported_gaps:
        if gap.requested_topic.lower() not in existing_refuse:
            req.must_refuse_topics.append(f"{gap.requested_topic} (unsupported by documents or tools)")
            existing_refuse.add(gap.requested_topic.lower())

    return req


def run_intake_interview(
    brief: str,
    manifest: CorpusManifest,
    llm: LLMClient,
    unattended: bool = False,
    answer_fn: Optional[Callable[[str, int, int], str]] = None,
) -> RequirementsDoc:
    """Inspect brief + corpus manifest + registered tools, ask <=5 questions, and output RequirementsDoc."""
    probe = probe_intake(brief=brief, manifest=manifest, llm=llm)
    questions = probe.clarifying_questions[:5]
    qa_pairs: list[ClarifyingQA] = []

    if not unattended and questions:
        for idx, q in enumerate(questions, start=1):
            if answer_fn is not None:
                ans = answer_fn(q, idx, len(questions))
            else:
                ans = input(f"[Clarifying Q{idx}/{len(questions)}] {q}\n> ").strip()
            qa_pairs.append(
                ClarifyingQA(
                    question=q,
                    answer=ans or "Use conservative, document- and tool-grounded defaults.",
                )
            )
    else:
        for q in questions:
            qa_pairs.append(
                ClarifyingQA(
                    question=q,
                    answer="Unattended mode: use conservative defaults grounded in the provided documents and any applicable registered tools.",
                )
            )

    return synthesize_requirements(
        brief=brief,
        manifest=manifest,
        probe=probe,
        qa_pairs=qa_pairs,
        llm=llm,
    )
