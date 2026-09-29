"""Dataset adapters for APIGen/xLAM and the selector slice of BFCL."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    text: str


@dataclass(frozen=True)
class DecisionExample:
    example_id: str
    query: str
    candidates: list[Candidate]
    target_weights: list[float]
    group_id: str
    source: str

    def validate(self) -> None:
        if not self.query.strip():
            raise ValueError(f"{self.example_id}: query is empty")
        if len(self.candidates) < 2:
            raise ValueError(f"{self.example_id}: at least two candidates are required")
        if len(self.candidates) != len(self.target_weights):
            raise ValueError(f"{self.example_id}: candidates/targets have different lengths")
        if any(weight < 0 for weight in self.target_weights):
            raise ValueError(f"{self.example_id}: target weights must be nonnegative")
        if abs(sum(self.target_weights) - 1.0) > 1e-5:
            raise ValueError(f"{self.example_id}: target weights must sum to one")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.example_id}: candidate IDs must be unique")
        if not any(weight > 0 for weight in self.target_weights):
            raise ValueError(f"{self.example_id}: at least one target must be positive")


def _decode_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _normalise_query(query: str) -> str:
    query = unicodedata.normalize("NFKC", query).casefold()
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def tool_id(tool: dict[str, Any]) -> str:
    name = tool.get("name") or tool.get("function_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool schema has no nonempty name")
    return name.strip()


def tool_text(tool: dict[str, Any]) -> str:
    """Create a stable text view that includes the schema's decision-relevant fields."""
    schema = {
        "name": tool.get("name") or tool.get("function_name"),
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", tool.get("parameters_schema", {})),
    }
    return "[ACTION] " + json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def query_text(query: str, state: str = "") -> str:
    if state:
        return f"[QUERY] State: {state}\nQuestion: {query}"
    return f"[QUERY] {query}"


def _xlam_record_to_example(record: dict[str, Any]) -> tuple[DecisionExample | None, str]:
    query = _decode_jsonish(record.get("query", ""))
    tools = _decode_jsonish(record.get("tools", []))
    answers = _decode_jsonish(record.get("answers", []))
    if not isinstance(query, str) or not isinstance(tools, list) or not isinstance(answers, list):
        return None, "bad_schema"
    if not query.strip():
        return None, "empty_query"

    try:
        unique_tools: dict[str, dict[str, Any]] = {}
        for tool in tools:
            if isinstance(tool, dict):
                unique_tools.setdefault(tool_id(tool), tool)
        target_ids = {
            str(answer.get("name", "")).strip()
            for answer in answers
            if isinstance(answer, dict) and str(answer.get("name", "")).strip()
        }
    except ValueError:
        return None, "bad_tool_schema"

    # The first experiment is exclusive single-tool selection. Parallel calls
    # are deliberately left for a later set-prediction extension.
    if len(target_ids) != 1:
        return None, "not_single_target"
    target_id = next(iter(target_ids))
    if target_id not in unique_tools:
        return None, "target_missing_from_candidates"
    if len(unique_tools) < 2:
        return None, "fewer_than_two_candidates"

    ordered_tools = sorted(unique_tools.items(), key=lambda pair: pair[0])
    candidates = [Candidate(name, tool_text(tool)) for name, tool in ordered_tools]
    weights = [1.0 if name == target_id else 0.0 for name, _ in ordered_tools]
    raw_id = str(record.get("id", "unknown"))
    group_id = hashlib.sha256(_normalise_query(query).encode("utf-8")).hexdigest()
    example = DecisionExample(
        example_id=f"xlam-{raw_id}",
        query=query_text(query),
        candidates=candidates,
        target_weights=weights,
        group_id=group_id,
        source="Salesforce/xlam-function-calling-60k",
    )
    example.validate()
    return example, "kept"


def read_jsonl(path: str | Path) -> list[DecisionExample]:
    examples: list[DecisionExample] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            candidates = [Candidate(**candidate) for candidate in row["candidates"]]
            example = DecisionExample(
                example_id=str(row["example_id"]),
                query=str(row["query"]),
                candidates=candidates,
                target_weights=[float(value) for value in row["target_weights"]],
                group_id=str(row["group_id"]),
                source=str(row.get("source", "unknown")),
            )
            try:
                example.validate()
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            examples.append(example)
    return examples


def write_jsonl(path: str | Path, examples: Iterable[DecisionExample]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for example in examples:
            example.validate()
            stream.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(destination)
    return count


def _split_for_group(group_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") % 100
    if value < 80:
        return "train"
    if value < 85:
        return "validation"
    if value < 90:
        return "calibration"
    return "test"


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _bm25_hard_negatives(
    examples: list[DecisionExample], count: int
) -> tuple[list[DecisionExample], dict[str, Any]]:
    """Append BM25-retrieved training-fold schemas as explicitly weak negatives."""
    if count < 0:
        raise ValueError("BM25 negative count cannot be negative")
    if count == 0 or not examples:
        return examples, {"method": "bm25", "requested_per_example": count, "added": 0}

    catalog: dict[str, Candidate] = {}
    positives_by_group: dict[str, set[str]] = {}
    for example in examples:
        positives_by_group.setdefault(example.group_id, set()).update(
            candidate.candidate_id
            for candidate, weight in zip(example.candidates, example.target_weights)
            if weight > 0
        )
        for candidate in example.candidates:
            catalog.setdefault(candidate.candidate_id, candidate)

    document_terms = {key: _tokens(candidate.text) for key, candidate in catalog.items()}
    document_frequency: dict[str, int] = {}
    for terms in document_terms.values():
        for term in set(terms):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    document_count = len(document_terms)
    average_length = sum(map(len, document_terms.values())) / max(document_count, 1)
    k1, b = 1.2, 0.75
    added_total = 0
    augmented: list[DecisionExample] = []

    for example in examples:
        existing = {candidate.candidate_id for candidate in example.candidates}
        protected = positives_by_group.get(example.group_id, set())
        query = _tokens(example.query.removeprefix("[QUERY] "))
        query_frequency: dict[str, int] = {}
        for term in query:
            query_frequency[term] = query_frequency.get(term, 0) + 1
        ranked: list[tuple[float, str]] = []
        for candidate_id, terms in document_terms.items():
            if candidate_id in existing or candidate_id in protected:
                continue
            term_frequency: dict[str, int] = {}
            for term in terms:
                term_frequency[term] = term_frequency.get(term, 0) + 1
            score = 0.0
            for term, query_count in query_frequency.items():
                frequency = term_frequency.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log1p(
                    (document_count - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0 - b + b * len(terms) / max(average_length, 1.0)
                )
                score += inverse_frequency * frequency * (k1 + 1.0) / denominator * query_count
            if score > 0:
                ranked.append((score, candidate_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = [catalog[candidate_id] for _, candidate_id in ranked[:count]]
        if selected:
            augmented.append(
                DecisionExample(
                    example_id=example.example_id,
                    query=example.query,
                    candidates=[*example.candidates, *selected],
                    target_weights=[*example.target_weights, *([0.0] * len(selected))],
                    group_id=example.group_id,
                    source=example.source,
                )
            )
            added_total += len(selected)
        else:
            augmented.append(example)

    return augmented, {
        "method": "BM25 over unique schemas from the training fold only",
        "requested_per_example": count,
        "added": added_total,
        "negative_label_caveat": "retrieved schemas are weak negatives and may include an unannotated valid action",
    }


def prepare_xlam_records(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    seed: int = 42,
    bm25_negatives: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[DecisionExample]] = {
        "train": [], "validation": [], "calibration": [], "test": []
    }
    skipped: dict[str, int] = {}
    seen_query_split: dict[str, str] = {}

    for record in records:
        example, reason = _xlam_record_to_example(record)
        skipped[reason] = skipped.get(reason, 0) + (example is None)
        if example is None:
            continue
        split = seen_query_split.setdefault(example.group_id, _split_for_group(example.group_id, seed))
        split_rows[split].append(example)

    counts = {
        split: 0 for split in split_rows
    }
    hard_negative_info: dict[str, Any] = {"method": "BM25", "requested_per_example": bm25_negatives, "added": 0}
    split_rows["train"], hard_negative_info = _bm25_hard_negatives(split_rows["train"], bm25_negatives)
    for split, rows in split_rows.items():
        counts[split] = write_jsonl(output / f"{split}.jsonl", rows)
    if any(counts[name] == 0 for name in ("train", "validation", "calibration", "test")):
        raise ValueError(f"prepared split is empty; inspect source rows and grouping: {counts}")
    manifest = {
        "dataset": "Salesforce/xlam-function-calling-60k",
        "dataset_license": "CC-BY-4.0; gated access requires the user to accept the dataset conditions",
        "selection_task": "single distinct target tool among supplied candidate schemas",
        "split_seed": seed,
        "split_by": "normalized query hash; API schemas may appear in multiple splits",
        "hard_negatives": hard_negative_info,
        "counts": counts,
        "skipped": skipped,
    }
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def download_and_prepare_xlam(
    output_dir: str | Path, seed: int = 42, bm25_negatives: int = 0
) -> dict[str, Any]:
    """Download APIGen data from its gated primary HF repository and normalize it."""
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "Salesforce/xlam-function-calling-60k"
    filename = "xlam_function_calling_60k.json"
    try:
        info = HfApi().dataset_info(repo_id)
        raw_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", revision=info.sha
        )
    except Exception as error:  # noqa: BLE001 - make gated-access failures actionable.
        raise RuntimeError(
            "Could not access the gated xLAM dataset. Log in to Hugging Face and accept the "
            "Salesforce dataset conditions in a browser, then provide HF_TOKEN through the "
            "environment on the remote preparation job. The token is never written to the project."
        ) from error

    raw_text = Path(raw_path).read_text(encoding="utf-8")
    try:
        raw_data = json.loads(raw_text)
        records = raw_data if isinstance(raw_data, list) else list(raw_data.values())
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("xLAM source file did not decode to a list of JSON records")

    manifest = prepare_xlam_records(records, output_dir, seed, bm25_negatives)
    manifest["source_revision"] = getattr(info, "sha", None)
    manifest["raw_file_sha256"] = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
    manifest_path = Path(output_dir) / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def _question_as_text(value: Any) -> str:
    value = _decode_jsonish(value)
    parts: list[str] = []

    def collect(item: Any) -> None:
        item = _decode_jsonish(item)
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
        elif isinstance(item, dict):
            if "content" in item:
                collect(item["content"])
            elif isinstance(item.get("text"), str):
                collect(item["text"])
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value)
    return "\n".join(parts)


def load_bfcl_category(data_path: str | Path, answers_path: str | Path) -> list[DecisionExample]:
    """Load BFCL single-tool-selection cases; exact argument generation is out of scope."""
    questions: dict[str, dict[str, Any]] = {}
    with Path(data_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                questions[str(row["id"])] = row
    answers: dict[str, dict[str, Any]] = {}
    with Path(answers_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                answers[str(row["id"])] = row

    examples: list[DecisionExample] = []
    for example_id, row in questions.items():
        answer = answers.get(example_id)
        if answer is None:
            continue
        function_docs = _decode_jsonish(row.get("function", []))
        ground_truth = _decode_jsonish(answer.get("ground_truth", []))
        if not isinstance(function_docs, list) or not isinstance(ground_truth, list):
            continue
        try:
            tool_map = {tool_id(tool): tool for tool in function_docs if isinstance(tool, dict)}
        except ValueError:
            continue
        target_ids = {
            str(name)
            for truth in ground_truth
            if isinstance(truth, dict)
            for name in truth.keys()
        }
        if len(target_ids) != 1 or not target_ids.issubset(tool_map) or len(tool_map) < 2:
            continue
        ordered = sorted(tool_map.items())
        candidates = [Candidate(name, tool_text(tool)) for name, tool in ordered]
        target = next(iter(target_ids))
        weights = [1.0 if name == target else 0.0 for name, _ in ordered]
        query = _question_as_text(row.get("question", ""))
        if not query.strip():
            continue
        example = DecisionExample(
            example_id=example_id,
            query=query_text(query),
            candidates=candidates,
            target_weights=weights,
            group_id=example_id,
            source="Berkeley-Function-Calling-Leaderboard",
        )
        example.validate()
        examples.append(example)
    return examples


def download_bfcl(output_dir: str | Path, revision: str = "main") -> tuple[Path, str]:
    """Fetch the public BFCL data snapshot used by the selector evaluator."""
    from huggingface_hub import HfApi, snapshot_download

    repo_id = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    resolved_revision = HfApi().dataset_info(repo_id, revision=revision).sha
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            local_dir=output_dir,
            allow_patterns=[
                "BFCL_v3_multiple.json",
                "BFCL_v3_live_multiple.json",
                "possible_answer/BFCL_v3_multiple.json",
                "possible_answer/BFCL_v3_live_multiple.json",
            ],
        )
    )
    return snapshot, resolved_revision
