"""Frozen, original-candidate external tool-selection slices (not official scores)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data import Candidate, DecisionExample, _normalise_query, query_text, read_jsonl, tool_id, tool_text, write_jsonl

SOURCES = {
    "when2call": ("nvidia/When2Call", ["test/when2call_test_llm_judge.jsonl", "test/when2call_test_mcq.jsonl"], "cc-by-4.0"),
    "toolace": ("Team-ACE/ToolACE", ["data.json"], "apache-2.0"),
}


def _example(query: str, tools: list[Any], target: str, source: str, identity: str) -> DecisionExample:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("empty_query")
    schemas = {}
    for tool in tools:
        tool = json.loads(tool) if isinstance(tool, str) else tool
        if not isinstance(tool, dict):
            raise ValueError("bad_tool_schema")
        if isinstance(tool.get("function"), dict):
            tool = tool["function"]
        name = tool_id(tool)
        if name in schemas:
            raise ValueError("duplicate_tool_names")
        schemas[name] = tool
    if len(schemas) < 2:
        raise ValueError("fewer_than_two_candidates")
    if target not in schemas:
        raise ValueError("target_missing_from_candidates")
    ordered = sorted(schemas.items())
    group = hashlib.sha256(_normalise_query(query).encode()).hexdigest()
    example = DecisionExample(identity, query_text(query), [Candidate(n, tool_text(t)) for n, t in ordered],
                              [float(n == target) for n, _ in ordered], group, source)
    example.validate()
    return example


def when2call_example(row: dict[str, Any], index: int) -> DecisionExample:
    if row.get("correct_answer") != "tool_call":
        raise ValueError("not_tool_call")
    answer = row.get("answers", {}).get("tool_call")
    answer = json.loads(answer) if isinstance(answer, str) else answer
    if not isinstance(answer, dict) or not isinstance(answer.get("name"), str):
        raise ValueError("bad_target")
    return _example(row.get("question"), row.get("tools", []), answer["name"],
                    SOURCES["when2call"][0], f"when2call-{row.get('uuid', index)}")


def _call_names(text: str) -> list[str]:
    """Strict bracketed top-level call syntax; never execute source expressions."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("not_explicit_call")
    body = text[1:-1].strip()
    names = []
    while body:
        opening = body.find("(")
        if opening <= 0:
            raise ValueError("bad_call_syntax")
        name = body[:opening].strip()
        stack, quote, escape, end = [], None, False, None
        for i, ch in enumerate(body[opening:], opening):
            if quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if not stack or stack.pop() != {")": "(", "]": "[", "}": "{"}[ch]:
                    raise ValueError("bad_call_syntax")
                if not stack:
                    if ch != ")":
                        raise ValueError("bad_call_syntax")
                    end = i + 1
                    break
        if end is None:
            raise ValueError("bad_call_syntax")
        names.append(name)
        body = body[end:].strip()
        if body:
            if not body.startswith(","):
                raise ValueError("bad_call_syntax")
            body = body[1:].strip()
            if not body:
                raise ValueError("bad_call_syntax")
    return names


def toolace_example(row: dict[str, Any], index: int) -> DecisionExample:
    messages = row.get("conversations", [])
    if len(messages) < 2 or messages[0].get("from") != "user" or messages[1].get("from") != "assistant":
        raise ValueError("not_first_user_assistant_pair")
    system = row.get("system", "")
    marker = "Here is a list of functions in JSON format that you can invoke:"
    if marker not in system:
        raise ValueError("missing_tools_marker")
    tools, _ = json.JSONDecoder().raw_decode(system.split(marker, 1)[1].lstrip())
    if not isinstance(tools, list):
        raise ValueError("bad_tool_schema")
    names = set(_call_names(messages[1].get("value", "")))
    if len(names) != 1:
        raise ValueError("not_single_target")
    return _example(messages[0].get("value"), tools, next(iter(names)), SOURCES["toolace"][0], f"toolace-{index}")


def prepare_external(output_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Download primary pinned files and exclude exact normalized xLAM queries."""
    from huggingface_hub import HfApi, hf_hub_download

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    xlam = Path(config["xlam_dir"])
    excluded = set()
    xlam_hashes = {}
    for split in ("train", "validation", "calibration", "test"):
        path = xlam / f"{split}.jsonl"
        xlam_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        excluded.update(_normalise_query(e.query.removeprefix("[QUERY] ")) for e in read_jsonl(path))
    manifest = {"metric_scope": "custom single-target tool-selection slices; no argument generation or official benchmark equivalence",
                "exclusion": "exact NFKC/casefold/whitespace normalized query overlap with all xLAM splits; not a semantic contamination guarantee",
                "xlam_sha256": xlam_hashes, "datasets": []}
    api = HfApi()
    for key, (repo, filenames, license_name) in SOURCES.items():
        info = api.dataset_info(repo)
        rows, raw_hashes = [], {}
        for filename in filenames:
            path = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=info.sha))
            raw = path.read_text(encoding="utf-8")
            file_rows = [json.loads(line) for line in raw.splitlines() if line.strip()] if filename.endswith("jsonl") else json.loads(raw)
            if not isinstance(file_rows, list):
                raise ValueError(f"{repo}: expected list of records")
            rows.extend(file_rows)
            raw_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        examples, seen, counts = [], set(), Counter()
        adapter = when2call_example if key == "when2call" else toolace_example
        for index, row in enumerate(rows):
            try:
                example = adapter(row, index)
                normalized = _normalise_query(example.query.removeprefix("[QUERY] "))
                if normalized in excluded:
                    raise ValueError("exact_xlam_query_overlap")
                if normalized in seen:
                    raise ValueError("duplicate_query_within_dataset")
                seen.add(normalized)
                examples.append(example)
                counts["kept"] += 1
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                reason = str(error) if isinstance(error, ValueError) and len(str(error)) < 80 else "bad_schema"
                counts[reason] += 1
        if not examples:
            raise ValueError(f"{repo}: no eligible examples: {dict(counts)}")
        destination = output / f"{key}.jsonl"
        write_jsonl(destination, examples)
        manifest["datasets"].append({"name": key, "path": str(destination.resolve()), "repository": repo, "source_url": f"https://huggingface.co/datasets/{repo}",
            "revision": info.sha, "source_revision": info.sha, "source_filenames": filenames, "license": license_name,
            "raw_sha256": raw_hashes, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source_rows": len(rows), "counts": dict(counts), "examples": len(examples), "candidate_policy": "original supplied candidates only; no synthetic negatives",
            "selection": "correct_answer=tool_call in test/llm_judge and test/mcq; normalized-query deduplication across both" if key == "when2call" else "first user/assistant pair only; one distinct explicit target",
            "caveat": "BFCL-derived queries; correlated stress evaluation, not independent of BFCL" if key == "when2call" else "synthetic training corpus repurposed as external evaluation; frozen models never trained on this corpus in this project"})
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    return manifest
