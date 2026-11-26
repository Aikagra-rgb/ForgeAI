"""Generic Agent Runner that executes any declarative AgentConfig."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from forgeai.ingestion import HybridRetriever, chunk_corpus
from forgeai.llm import LLMClient
from forgeai.schemas import AgentConfig, AgentTrace, DocumentChunk, ToolCallTrace
from forgeai.tools import execute_tool, get_openai_tools


def _extract_cited_sources(answer: str, retrieved: list[DocumentChunk]) -> list[str]:
    """Detect and normalize source filenames cited in the answer."""
    cited: set[str] = set()
    candidate_names = {c.filename for c in retrieved}
    candidate_by_lower = {name.lower(): name for name in candidate_names}
    answer_lower = answer.lower()

    for name in candidate_names:
        stem = name.rsplit(".", 1)[0]
        if name.lower() in answer_lower or stem.lower() in answer_lower:
            cited.add(name)

    # Catch explicit [Source: ...] patterns, splitting comma lists and stripping "| chunk N"
    for match in re.findall(r"\[Source:\s*([^\]]+)\]", answer, flags=re.IGNORECASE):
        for part in match.split(","):
            head = part.split("|", 1)[0].strip()
            if not head:
                continue
            cited.add(candidate_by_lower.get(head.lower(), head))
    return sorted(cited)


class GenericAgentRunner:
    """Runs any produced AgentConfig against a document folder."""

    def __init__(
        self,
        config: AgentConfig,
        docs_dir: str,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.docs_dir = docs_dir
        self.llm = llm or LLMClient()
        self.chunks = chunk_corpus(
            docs_dir=docs_dir,
            chunk_size=config.retrieval_config.chunk_size,
            chunk_overlap=config.retrieval_config.chunk_overlap,
        )
        self.retriever = HybridRetriever(self.chunks)
        self.selected_tool_names = [t.name for t in config.selected_tools]
        self.openai_tools = get_openai_tools(self.selected_tool_names)

    def _build_system_message(self) -> str:
        base = self.config.system_prompt.render()
        contract = self.config.output_contract
        contract_lines = [
            f"## Output Contract",
            f"- Format: {contract.format}",
            f"- Citations required: {contract.require_citations} ({contract.citation_format})",
        ]
        for rule in contract.additional_rules:
            contract_lines.append(f"- {rule}")
        return base + "\n\n" + "\n".join(contract_lines)

    def run(
        self,
        question: str,
        history: Optional[list[dict[str, str]]] = None,
        max_tool_rounds: int = 3,
    ) -> AgentTrace:
        """Answer a single user query using retrieval + only the configured tools."""
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(
            query=question,
            top_k=self.config.retrieval_config.top_k,
            mode=self.config.retrieval_config.retrieval_mode,
        )

        context_blocks = [
            f"[Source: {c.filename} | {c.page_or_row}]\n{c.text}" for c in retrieved
        ]
        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "(No matching document chunks retrieved.)"

        messages: list[dict[str, Any]] = [{"role": "system", "content": self._build_system_message()}]
        if history:
            messages.extend(history[-4:])

        user_payload = (
            f"Retrieved Context:\n{context_str}\n\n"
            f"User Question:\n{question}\n\n"
            "Answer the current User Question directly and strictly based on the retrieved context and any tool outputs. "
            "Do not carry over unrelated numbers or items from previous chat turns unless the user explicitly refers to them. "
            "Carefully check numeric thresholds and exception clauses (such as 'unless', 'more than', or 'under'). "
            "Cite source filenames using [Source: <filename>]. "
            "If the retrieved context contains relevant facts, answer directly; "
            "only refuse when the question is out-of-scope or completely unsupported by the context."
        )
        messages.append({"role": "user", "content": user_payload})

        total_tokens = 0
        tool_traces: list[ToolCallTrace] = []
        final_answer = ""

        for _ in range(max_tool_rounds + 1):
            msg, tokens, _ = self.llm.chat(
                messages=messages,
                temperature=0.1,
                tools=self.openai_tools if self.openai_tools else None,
            )
            total_tokens += tokens

            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                final_answer = (msg.content or "").strip()
                break

            # Append assistant tool-call message
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = execute_tool(fn_name, args, docs_dir=self.docs_dir)
                tool_traces.append(
                    ToolCallTrace(tool_name=fn_name, arguments=args, result=result)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        else:
            final_answer = (msg.content or "").strip()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        cited = _extract_cited_sources(final_answer, retrieved)

        return AgentTrace(
            question=question,
            answer=final_answer,
            retrieved_chunks=retrieved,
            tool_calls=tool_traces,
            cited_sources=cited,
            latency_ms=latency_ms,
            tokens_used=total_tokens,
        )
