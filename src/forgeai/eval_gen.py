"""Chunk-grounded Evaluation Set Generator (>=20 cases) & Fact Validation."""

from __future__ import annotations

import random
import re
from typing import Optional

from pydantic import BaseModel, Field

from forgeai.ingestion import chunk_corpus
from forgeai.llm import LLMClient
from forgeai.schemas import (
    DocumentChunk,
    EvalCase,
    EvalCaseValidation,
    EvalSet,
    EvalValidationReport,
    FactValidationResult,
    RequirementsDoc,
)

ANSWERABLE_SYSTEM_PROMPT = (
    "You are ForgeAI's Grounded Eval Generator. "
    "Given ONE exact document chunk, write a standalone realistic user question that can ONLY be answered "
    "using this chunk, plus 1-3 concise atomic `expected_facts` directly stated in the chunk. "
    "Rules: (1) Each `expected_fact` MUST be a complete concrete factual statement or value (NEVER a question "
    "fragment or placeholder like 'who approves its use'). "
    "(2) `expected_facts` must strictly answer ONLY what the question asks using the exact matching column/field—"
    "never include unasked columns or pick an empty CSV field (if a CSV cell like `panamaFee` is empty, do not ask about it). "
    "(3) If a rule in the chunk applies only to a specific subgroup (e.g., airline flight crew, key employees, "
    "a specific vessel class), explicitly name that subgroup in the question. "
    "(4) NEVER write 'Which of the following', 'in this document', or 'this policy'—name the policy/program/entity. "
    "(5) Do NOT invent facts outside the chunk."
)

REFUSAL_SYSTEM_PROMPT = (
    "You are ForgeAI's Out-of-Scope Eval Generator. "
    "Given an agent's scope, `must_answer_topics`, `tool_supported_capabilities`, and `must_refuse_topics`, "
    "generate tempting, domain-adjacent user questions that MUST be refused because they fall in `must_refuse_topics` "
    "or require information/actions absent from BOTH the documents and tools. "
    "When `must_refuse_topics` is non-empty, generate at most 2 questions per single refused topic (so a 1-item list "
    "does NOT produce 5 near-duplicate questions); fill remaining slots with clearly distinct out-of-scope topics "
    "absent from BOTH documents and tools (e.g., legal representation, external email drafting, tax filing, "
    "live external API actions, or unrelated policies). "
    "NEVER generate a refusal question that is answerable by `must_answer_topics`, `tool_supported_capabilities`, "
    "or by summarizing/listing records present in the documents or CSVs."
)

MULTI_DOC_SYSTEM_PROMPT = (
    "You are ForgeAI's Multi-Document Edge-Case Generator. "
    "Given TWO chunks from different documents (or distant sections), write ONE realistic "
    "question that requires synthesizing facts from BOTH chunks, and list 2-3 concise atomic `expected_facts` "
    "explicitly stated in Chunk A and Chunk B. Do NOT extrapolate or invent unstated comparisons."
)

AMBIGUOUS_SYSTEM_PROMPT = (
    "You are ForgeAI's Ambiguity Edge-Case Generator. "
    "Given ONE reference chunk and agent scope, write an intentionally underspecified or ambiguous "
    "user question and list 1-2 concise `expected_facts` directly stated in the reference chunk "
    "(such as key policy constraints or parameters that depend on missing user details). "
    "NEVER invent external policies, agreements, or fields not explicitly present in the chunk."
)

GENERATION_PROMPTS = {
    "answerable": ANSWERABLE_SYSTEM_PROMPT,
    "refusal": REFUSAL_SYSTEM_PROMPT,
    "edge_multi_doc": MULTI_DOC_SYSTEM_PROMPT,
    "edge_ambiguous": AMBIGUOUS_SYSTEM_PROMPT,
}


class _SingleQA(BaseModel):
    question: str
    expected_facts: list[str] = Field(min_length=1, max_length=4)
    rationale: str = ""


class _RefusalList(BaseModel):
    questions: list[str] = Field(min_length=3, max_length=7)


class _FactSupportBatch(BaseModel):
    results: list[FactValidationResult] = Field(default_factory=list)


class _RefusalValidityItem(BaseModel):
    question: str
    truly_out_of_scope: bool
    reason: str = ""


class _RefusalValidityBatch(BaseModel):
    results: list[_RefusalValidityItem] = Field(default_factory=list)


class _SingleRefusal(BaseModel):
    question: str


def _sample_chunks_balanced(chunks: list[DocumentChunk], count: int, seed: int = 42) -> list[DocumentChunk]:
    if not chunks:
        return []
    rng = random.Random(seed)
    orig_by_file: dict[str, list[DocumentChunk]] = {}
    for c in chunks:
        orig_by_file.setdefault(c.filename, []).append(c)

    by_file = {k: list(v) for k, v in orig_by_file.items()}
    selected: list[DocumentChunk] = []
    filenames = list(by_file.keys())
    rng.shuffle(filenames)

    idx = 0
    while len(selected) < count:
        if not filenames:
            # Small corpus fallback: refill pools so >=20 eval cases are always generated
            by_file = {k: list(v) for k, v in orig_by_file.items()}
            filenames = list(by_file.keys())
            rng.shuffle(filenames)
        fname = filenames[idx % len(filenames)]
        pool = by_file[fname]
        if pool:
            choice = rng.choice(pool)
            pool.remove(choice)
            selected.append(choice)
        else:
            filenames.remove(fname)
            continue
        idx += 1
    return selected


def _generate_eval_set_from_chunks(
    requirements: RequirementsDoc,
    chunks: list[DocumentChunk],
    llm: LLMClient,
    min_cases: int = 20,
    seed: int = 42,
) -> EvalSet:
    if not chunks:
        raise ValueError("No chunks available for eval generation")

    n_answerable = max(10, min_cases - 10)
    n_refusal = 5
    n_multi = 3
    n_ambiguous = 2

    cases: list[EvalCase] = []

    # 1. Answerable cases (chunk-first)
    sampled = _sample_chunks_balanced(chunks, n_answerable, seed=seed)
    for i, chunk in enumerate(sampled, start=1):
        qa = llm.structured_chat(
            messages=[
                {"role": "system", "content": ANSWERABLE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Source file: {chunk.filename} ({chunk.page_or_row})\n"
                        f"Chunk text:\n{chunk.text}"
                    ),
                },
            ],
            response_model=_SingleQA,
            temperature=0.2,
        )
        cases.append(
            EvalCase(
                id=f"ans_{i:02d}",
                category="answerable",
                question=qa.question,
                expected_source_docs=[chunk.filename],
                expected_facts=qa.expected_facts,
                should_refuse=False,
                source_chunk_ids=[chunk.chunk_id],
                rationale=qa.rationale or f"Grounded in {chunk.filename} ({chunk.page_or_row})",
            )
        )

    # 2. Out-of-scope refusal cases
    refusal_parsed = llm.structured_chat(
        messages=[
            {"role": "system", "content": REFUSAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Agent Scope: {requirements.scope}\n"
                    f"Must-Answer Topics (DO NOT put in refusal questions): {requirements.must_answer_topics}\n"
                    f"Tool-Supported Capabilities (DO NOT put in refusal questions): {requirements.tool_supported_capabilities}\n"
                    f"Must-Refuse Topics: {requirements.must_refuse_topics}\n"
                    f"Generate {n_refusal} distinct out-of-scope questions."
                ),
            },
        ],
        response_model=_RefusalList,
        temperature=0.3,
    )
    for i, q in enumerate(refusal_parsed.questions[:n_refusal], start=1):
        cases.append(
            EvalCase(
                id=f"ref_{i:02d}",
                category="refusal",
                question=q,
                expected_source_docs=[],
                expected_facts=["Politely refuse or state that the topic is out of scope / unsupported."],
                should_refuse=True,
                source_chunk_ids=[],
                rationale="Out-of-scope / unsupported topic refusal check",
            )
        )

    # 3. Multi-doc synthesis edge cases
    by_file: dict[str, list[DocumentChunk]] = {}
    for c in chunks:
        by_file.setdefault(c.filename, []).append(c)
    files = list(by_file.keys())
    rng = random.Random(seed + 1)

    for i in range(1, n_multi + 1):
        if len(files) >= 2:
            f1, f2 = rng.sample(files, 2)
            c1 = rng.choice(by_file[f1])
            c2 = rng.choice(by_file[f2])
        else:
            pool = by_file[files[0]]
            c1 = pool[0]
            c2 = pool[-1]

        qa = llm.structured_chat(
            messages=[
                {"role": "system", "content": MULTI_DOC_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Chunk A ({c1.filename}):\n{c1.text}\n\n"
                        f"Chunk B ({c2.filename}):\n{c2.text}"
                    ),
                },
            ],
            response_model=_SingleQA,
            temperature=0.2,
        )
        cases.append(
            EvalCase(
                id=f"multi_{i:02d}",
                category="edge_multi_doc",
                question=qa.question,
                expected_source_docs=sorted({c1.filename, c2.filename}),
                expected_facts=qa.expected_facts,
                should_refuse=False,
                source_chunk_ids=[c1.chunk_id, c2.chunk_id],
                rationale=qa.rationale or f"Cross-document synthesis ({c1.filename} + {c2.filename})",
            )
        )

    # 4. Ambiguous phrasing edge cases
    amb_chunks = _sample_chunks_balanced(chunks, n_ambiguous, seed=seed + 2)
    for i, chunk in enumerate(amb_chunks, start=1):
        qa = llm.structured_chat(
            messages=[
                {"role": "system", "content": AMBIGUOUS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Agent Scope: {requirements.scope}\n"
                        f"Reference Chunk ({chunk.filename}):\n{chunk.text}"
                    ),
                },
            ],
            response_model=_SingleQA,
            temperature=0.3,
        )
        cases.append(
            EvalCase(
                id=f"amb_{i:02d}",
                category="edge_ambiguous",
                question=qa.question,
                expected_source_docs=[chunk.filename],
                expected_facts=qa.expected_facts,
                should_refuse=False,
                source_chunk_ids=[chunk.chunk_id],
                rationale=qa.rationale or "Ambiguous/underspecified query handling",
            )
        )

    return EvalSet(agent_name=requirements.agent_name, cases=cases)


def _verify_facts_against_chunks(
    question: str,
    facts: list[str],
    cited_text: str,
    llm: LLMClient,
) -> tuple[list[str], list[str]]:
    """Return (kept_facts, dropped_facts) verified strictly against cited_text and the question."""
    if not facts:
        return [], []
    facts_block = "\n".join(f"{idx + 1}. {f}" for idx, f in enumerate(facts))
    batch = llm.structured_chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are ForgeAI's Strict Grounding Validator. "
                    "For EACH candidate `expected_fact`, verify BOTH:\n"
                    "1. `supported`: true ONLY if directly and explicitly stated in Cited Source Text (no invented details or assumptions).\n"
                    "2. `answers_question`: true ONLY if it directly answers the specific `Question` with a concrete fact "
                    "(false if it answers a different column/metric like `suezFee` when `panamaFee` was asked, or if it is "
                    "merely a question fragment/placeholder like 'who approves its use').\n"
                    "Return one `FactValidationResult` per fact in the exact same order."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nCited Source Text:\n{cited_text}\n\nCandidate Facts:\n{facts_block}",
            },
        ],
        response_model=_FactSupportBatch,
        temperature=0.0,
    )

    kept: list[str] = []
    dropped: list[str] = []

    if len(batch.results) == len(facts):
        for orig_fact, res in zip(facts, batch.results):
            if res.supported and res.answers_question:
                kept.append(orig_fact)
            else:
                dropped.append(orig_fact)
    else:
        res_map = {r.fact.strip().lower(): (r.supported and r.answers_question) for r in batch.results}
        for orig_fact in facts:
            if res_map.get(orig_fact.strip().lower(), False):
                kept.append(orig_fact)
            else:
                dropped.append(orig_fact)

    return kept, dropped


_FALLBACK_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "for", "are", "was", "were", "and", "or", "does", "did", "with",
    "from", "this", "that", "into", "have", "has", "can", "could", "would",
    "should", "about", "many", "much", "port", "leave", "total", "cost",
    "used", "using", "give", "tell", "list", "coordinates",
}


def _fallback_verbatim_fact(cited_chunks: list[DocumentChunk], question: str = "") -> str:
    """Extract the highest-overlap line/sentence from cited chunks matching the question."""
    candidates: list[str] = []
    for chunk in cited_chunks:
        for line in chunk.text.splitlines():
            clean_line = " ".join(line.split())
            if not clean_line:
                continue
            if clean_line.startswith("Row ") and "|" in clean_line:
                candidates.append(clean_line)
            else:
                for sent in re.split(r"(?<=[.!?])\s+", clean_line):
                    s = sent.strip()
                    if len(s) >= 15:
                        candidates.append(s)
                if len(clean_line) >= 15 and clean_line not in candidates:
                    candidates.append(clean_line)

    if not candidates:
        return " ".join(cited_chunks[0].text.split())[:320]

    q_tokens = {
        tok for tok in re.findall(r"[a-z0-9_]+", question.lower())
        if len(tok) >= 3 and tok not in _FALLBACK_STOPWORDS
    }
    proper_tokens = {
        tok.lower() for tok in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", question)
    }

    best_score = -1
    best_cand = candidates[0]
    for cand in candidates:
        c_tokens = set(re.findall(r"[a-z0-9_]+", cand.lower()))
        overlap = len(q_tokens & c_tokens)
        proper_bonus = len(proper_tokens & c_tokens)
        score = overlap * 2 + proper_bonus
        if score > best_score:
            best_score = score
            best_cand = cand

    return best_cand[:320]


def _validate_refusal_cases(
    refusal_cases: list[EvalCase],
    requirements: RequirementsDoc,
    llm: LLMClient,
) -> dict[str, EvalCase]:
    """Check refusal questions against must_answer/tools and replace any accidental in-scope questions."""
    if not refusal_cases:
        return {}

    q_block = "\n".join(f"{i + 1}. {c.question}" for i, c in enumerate(refusal_cases))
    batch = llm.structured_chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are ForgeAI's Refusal Sanity Checker. "
                    "Given an agent's `must_answer_topics`, `tool_supported_capabilities`, and `must_refuse_topics`, "
                    "determine whether EACH candidate refusal question is `truly_out_of_scope` (true ONLY if it cannot "
                    "be answered from the documents/must_answer_topics or supported tools; false if it is actually "
                    "in-scope or answerable, such as asking about adoption/parental leave in an HR leave guide). "
                    "Return one item per question in the exact same order."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scope: {requirements.scope}\n"
                    f"Must-Answer Topics: {requirements.must_answer_topics}\n"
                    f"Tool-Supported Capabilities: {requirements.tool_supported_capabilities}\n"
                    f"Must-Refuse Topics: {requirements.must_refuse_topics}\n\n"
                    f"Candidate Refusal Questions:\n{q_block}"
                ),
            },
        ],
        response_model=_RefusalValidityBatch,
        temperature=0.0,
    )

    updated: dict[str, EvalCase] = {}
    used_questions = [c.question for c in refusal_cases]

    for idx, case in enumerate(refusal_cases):
        if idx < len(batch.results):
            item = batch.results[idx]
        else:
            item = next((r for r in batch.results if r.question.strip().lower() == case.question.strip().lower()), None)
        if item is not None and not item.truly_out_of_scope:
            rep = llm.structured_chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate ONE replacement out-of-scope refusal question that CANNOT be answered by "
                            "`must_answer_topics` or `tool_supported_capabilities` and is distinct from already used questions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Scope: {requirements.scope}\n"
                            f"Must-Answer Topics (DO NOT ask about these): {requirements.must_answer_topics}\n"
                            f"Tool-Supported Capabilities (DO NOT ask about these): {requirements.tool_supported_capabilities}\n"
                            f"Must-Refuse Topics: {requirements.must_refuse_topics}\n"
                            f"Already Used Questions: {used_questions}\n"
                            f"Previous invalid question: {case.question} ({item.reason})"
                        ),
                    },
                ],
                response_model=_SingleRefusal,
                temperature=0.2,
            )
            used_questions.append(rep.question)
            updated[case.id] = case.model_copy(update={"question": rep.question})
        else:
            updated[case.id] = case

    return updated


def validate_eval_set(
    eval_set: EvalSet,
    chunks_by_id: dict[str, DocumentChunk],
    llm: LLMClient,
    requirements: Optional[RequirementsDoc] = None,
) -> tuple[EvalSet, EvalValidationReport]:
    """Validate non-refusal facts against cited chunks and sanity-check refusal cases."""
    refusal_replacements: dict[str, EvalCase] = {}
    if requirements is not None:
        refusal_cases = [c for c in eval_set.cases if c.should_refuse]
        refusal_replacements = _validate_refusal_cases(refusal_cases, requirements, llm)

    validated_cases: list[EvalCase] = []
    case_details: list[EvalCaseValidation] = []
    total_non_refusal = 0
    total_facts_checked = 0
    facts_kept_total = 0
    facts_dropped_total = 0
    cases_regenerated = 0

    for case in eval_set.cases:
        if case.should_refuse:
            validated_cases.append(refusal_replacements.get(case.id, case))
            continue

        total_non_refusal += 1
        cited_chunks = [chunks_by_id[cid] for cid in case.source_chunk_ids if cid in chunks_by_id]
        chunk_ids_valid = bool(case.source_chunk_ids) and len(cited_chunks) == len(case.source_chunk_ids) and all(
            c.filename in case.expected_source_docs for c in cited_chunks
        )

        if not cited_chunks:
            validated_cases.append(case)
            case_details.append(
                EvalCaseValidation(
                    case_id=case.id,
                    chunk_ids_valid=False,
                    original_fact_count=len(case.expected_facts),
                    kept_fact_count=0,
                    dropped_facts=list(case.expected_facts),
                    regenerated=False,
                )
            )
            total_facts_checked += len(case.expected_facts)
            facts_dropped_total += len(case.expected_facts)
            continue

        cited_text = "\n\n".join(
            f"[{c.filename} | {c.page_or_row}]\n{c.text}" for c in cited_chunks
        )
        orig_count = len(case.expected_facts)
        total_facts_checked += orig_count

        kept, dropped = _verify_facts_against_chunks(case.question, case.expected_facts, cited_text, llm)
        regenerated = False

        if not kept:
            # Regenerate once from the cited chunk(s) at temperature=0.0 and re-verify
            regenerated = True
            cases_regenerated += 1
            sys_prompt = GENERATION_PROMPTS.get(case.category, ANSWERABLE_SYSTEM_PROMPT)
            scope_line = f"Agent Scope: {requirements.scope}\n" if requirements else ""
            regen_qa = llm.structured_chat(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"{scope_line}Cited Source Text:\n{cited_text}"},
                ],
                response_model=_SingleQA,
                temperature=0.0,
            )
            case = case.model_copy(
                update={
                    "question": regen_qa.question,
                    "expected_facts": regen_qa.expected_facts,
                    "rationale": regen_qa.rationale or case.rationale,
                }
            )
            total_facts_checked += len(case.expected_facts)
            kept, dropped2 = _verify_facts_against_chunks(case.question, case.expected_facts, cited_text, llm)
            dropped = dropped + dropped2
            if not kept:
                kept = [_fallback_verbatim_fact(cited_chunks, case.question)]

        facts_kept_total += len(kept)
        facts_dropped_total += len(dropped)
        updated_case = case.model_copy(update={"expected_facts": kept})
        validated_cases.append(updated_case)
        case_details.append(
            EvalCaseValidation(
                case_id=updated_case.id,
                chunk_ids_valid=chunk_ids_valid,
                original_fact_count=orig_count,
                kept_fact_count=len(kept),
                dropped_facts=dropped,
                regenerated=regenerated,
            )
        )

    report = EvalValidationReport(
        total_non_refusal_cases=total_non_refusal,
        cases_validated=len(case_details),
        total_facts_checked=total_facts_checked,
        facts_kept=facts_kept_total,
        facts_dropped=facts_dropped_total,
        cases_regenerated=cases_regenerated,
        case_details=case_details,
    )
    return EvalSet(agent_name=eval_set.agent_name, cases=validated_cases), report


def generate_eval_set(
    requirements: RequirementsDoc,
    docs_dir: str,
    llm: LLMClient,
    min_cases: int = 20,
    seed: int = 42,
) -> EvalSet:
    """Generate >=20 chunk-grounded evaluation cases."""
    chunks = chunk_corpus(docs_dir, chunk_size=650, chunk_overlap=120)
    return _generate_eval_set_from_chunks(requirements, chunks, llm, min_cases=min_cases, seed=seed)


def generate_and_validate_eval_set(
    requirements: RequirementsDoc,
    docs_dir: str,
    llm: LLMClient,
    min_cases: int = 20,
    seed: int = 42,
) -> tuple[EvalSet, EvalValidationReport, EvalSet]:
    """Generate >=20 chunk-grounded cases, validate every non-refusal fact, and return both validated and raw eval sets.
    
    Returns:
        validated_eval: EvalSet with facts validated/dropped/replaced against chunks
        report: EvalValidationReport with stats on what was kept/dropped/regenerated
        raw_eval: EvalSet before validation (preserved for reviewer to judge validation impact)
    """
    chunks = chunk_corpus(docs_dir, chunk_size=650, chunk_overlap=120)
    raw_eval = _generate_eval_set_from_chunks(requirements, chunks, llm, min_cases=min_cases, seed=seed)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    validated_eval, report = validate_eval_set(raw_eval, chunks_by_id, llm, requirements=requirements)
    # Preserve raw eval with original generation prompts for reviewer analysis
    return validated_eval, report, raw_eval
