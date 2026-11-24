"""Tool registry for generated ForgeAI agents (4 focused tools)."""

from __future__ import annotations

import ast
import csv
import datetime
import json
import operator
from pathlib import Path
from typing import Any, Callable

import httpx

_SAFE_OPERATORS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    "round": round,
    "min": min,
    "max": max,
    "abs": abs,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn_name = node.func.id
        if fn_name in _SAFE_FUNCS:
            args = [_eval_ast(a) for a in node.args]
            return float(_SAFE_FUNCS[fn_name](*args))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def tool_calculator(expression: str, **_kwargs: Any) -> str:
    """Safely evaluate an arithmetic expression."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast(tree)
        if result.is_integer():
            return str(int(result))
        return f"{result:.6g}"
    except Exception as exc:
        return f"Calculator error: {exc}"


def tool_current_date(**_kwargs: Any) -> str:
    """Return the current local and UTC date/time."""
    now_local = datetime.datetime.now().astimezone()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    return json.dumps(
        {
            "local_date": now_local.strftime("%Y-%m-%d"),
            "local_time": now_local.strftime("%H:%M:%S %Z"),
            "utc_iso": now_utc.isoformat(),
        }
    )


def detect_csv_delimiter(path: Path) -> str:
    """Detect the delimiter for CSV/TSV files (handling tab-separated .csv files cleanly)."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(4096)
    first_line = next((line for line in sample.splitlines() if line.strip()), "")
    if "\t" in first_line and first_line.count("\t") >= first_line.count(","):
        return "\t"
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def tool_csv_lookup(
    query: str,
    filename: str = "",
    column: str = "",
    limit: int = 8,
    docs_dir: str = "",
    **_kwargs: Any,
) -> str:
    """Search CSV files in docs_dir for rows matching query tokens."""
    if not docs_dir:
        return "CSV lookup error: docs_dir not configured."

    base = Path(docs_dir)
    csv_files = sorted(base.rglob("*.csv"))
    if filename:
        csv_files = [p for p in csv_files if p.name.lower() == Path(filename).name.lower()]

    if not csv_files:
        return f"No matching CSV files found in {docs_dir}."

    tokens = [t.lower() for t in query.split() if t.strip()]

    def _search(use_column: bool) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for csv_path in csv_files:
            delimiter = detect_csv_delimiter(csv_path)
            with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                if reader.fieldnames:
                    reader.fieldnames = [col.strip() if col else col for col in reader.fieldnames]
                for row_idx, row in enumerate(reader, start=1):
                    clean_row = {
                        (k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v)
                        for k, v in row.items()
                    }
                    if use_column and column and column in clean_row:
                        haystack = str(clean_row[column]).lower()
                    else:
                        haystack = " ".join(str(v) for v in clean_row.values()).lower()
                    if not tokens or all(tok in haystack for tok in tokens) or any(tok in haystack for tok in tokens):
                        score = sum(1 for tok in tokens if tok in haystack)
                        found.append(
                            {
                                "_score": score,
                                "file": csv_path.name,
                                "row": row_idx,
                                "data": clean_row,
                            }
                        )
        return found

    matches = _search(use_column=bool(column))
    if column and not matches:
        matches = _search(use_column=False)

    matches.sort(key=lambda m: m["_score"], reverse=True)
    top = [{k: v for k, v in m.items() if k != "_score"} for m in matches[: max(1, min(limit, 20))]]
    if not top:
        return f"No CSV rows matched query '{query}'."
    return json.dumps(top, ensure_ascii=False)


def tool_open_meteo_weather(
    latitude: float,
    longitude: float,
    location_name: str = "",
    **_kwargs: Any,
) -> str:
    """Fetch current weather and 3-day forecast from Open-Meteo (keyless public API)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,wind_speed_10m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 3,
        "timezone": "auto",
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        return json.dumps(
            {
                "location": location_name or f"{latitude},{longitude}",
                "current": payload.get("current", {}),
                "daily": payload.get("daily", {}),
            }
        )
    except Exception as exc:
        return f"Open-Meteo API error: {exc}"


TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "csv_lookup": {
        "description": "Search structured CSV files in the knowledge folder by keyword or column.",
        "fn": tool_csv_lookup,
        "schema": {
            "type": "function",
            "function": {
                "name": "csv_lookup",
                "description": "Search CSV files in the document folder for rows matching a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords (e.g., part number, SKU, item name)."},
                        "filename": {"type": "string", "description": "Optional specific CSV filename."},
                        "column": {"type": "string", "description": "Optional column name to restrict search."},
                        "limit": {"type": "integer", "description": "Max rows to return (default 8)."},
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "calculator": {
        "description": "Evaluate exact arithmetic expressions (totals, percentages, unit conversions, lead times).",
        "fn": tool_calculator,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Safely evaluate an arithmetic expression such as '(120 * 4) * 1.15'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Arithmetic expression to compute."},
                    },
                    "required": ["expression"],
                },
            },
        },
    },
    "open_meteo_weather": {
        "description": "Fetch live weather and 3-day forecast for a latitude/longitude via Open-Meteo.",
        "fn": tool_open_meteo_weather,
        "schema": {
            "type": "function",
            "function": {
                "name": "open_meteo_weather",
                "description": "Get current weather and 3-day forecast for given coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "description": "Latitude in decimal degrees."},
                        "longitude": {"type": "number", "description": "Longitude in decimal degrees."},
                        "location_name": {"type": "string", "description": "Optional human-readable location name."},
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        },
    },
    "current_date": {
        "description": "Get the current local and UTC date and time.",
        "fn": tool_current_date,
        "schema": {
            "type": "function",
            "function": {
                "name": "current_date",
                "description": "Return the current local date, time, and UTC ISO timestamp.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
}


def format_tool_catalog() -> str:
    """Format the registered tool catalog with names, capabilities, and input requirements."""
    lines: list[str] = []
    for name, meta in TOOL_DEFINITIONS.items():
        params = meta["schema"]["function"]["parameters"].get("properties", {})
        required = set(meta["schema"]["function"]["parameters"].get("required", []))
        if params:
            param_desc = ", ".join(
                f"{p}{'*' if p in required else ''}: {spec.get('type', 'any')}"
                for p, spec in params.items()
            )
        else:
            param_desc = "none"
        lines.append(f"- `{name}`: {meta['description']} (inputs: {param_desc})")
    return "\n".join(lines)


def get_openai_tools(selected_names: list[str]) -> list[dict[str, Any]]:
    """Return OpenAI tool schemas for the selected tool names only."""
    return [TOOL_DEFINITIONS[name]["schema"] for name in selected_names if name in TOOL_DEFINITIONS]


def execute_tool(name: str, arguments: dict[str, Any], docs_dir: str = "") -> str:
    """Execute a registered tool safely and return a string result."""
    entry = TOOL_DEFINITIONS.get(name)
    if not entry:
        return f"Unknown tool: {name}"
    try:
        return str(entry["fn"](**arguments, docs_dir=docs_dir))
    except Exception as exc:
        return f"Tool '{name}' execution error: {exc}"
