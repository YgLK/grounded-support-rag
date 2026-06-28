"""Kubernetes website documentation loader and fetch helper."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from support_graph.types import Document

KUBERNETES_WEBSITE_REPO = "https://github.com/kubernetes/website.git"
KUBERNETES_DOCS_SUBPATH = Path("content/en/docs")

Runner = Callable[..., subprocess.CompletedProcess]


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    runner: Runner,
) -> subprocess.CompletedProcess:
    return runner(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def fetch_kubernetes_docs(
    output_dir: str | Path,
    *,
    ref: str = "main",
    replace: bool = False,
    repo_url: str = KUBERNETES_WEBSITE_REPO,
    runner: Runner = subprocess.run,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch a sparse Kubernetes website docs snapshot into a local corpus dir."""

    output_path = Path(output_dir)
    if output_path.exists():
        if not replace:
            raise FileExistsError(
                f"Kubernetes docs output already exists: {output_path}. Use --replace to overwrite."
            )
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="kubernetes-website-", dir=output_path.parent
    ) as temp_dir:
        checkout = Path(temp_dir)
        _run_git(["init"], cwd=checkout, runner=runner)
        _run_git(["remote", "add", "origin", repo_url], cwd=checkout, runner=runner)
        _run_git(["sparse-checkout", "init", "--cone"], cwd=checkout, runner=runner)
        _run_git(
            ["sparse-checkout", "set", str(KUBERNETES_DOCS_SUBPATH)],
            cwd=checkout,
            runner=runner,
        )
        _run_git(["fetch", "--depth", "1", "origin", ref], cwd=checkout, runner=runner)
        _run_git(["checkout", "--detach", "FETCH_HEAD"], cwd=checkout, runner=runner)
        resolved = _run_git(["rev-parse", "HEAD"], cwd=checkout, runner=runner)
        resolved_sha = resolved.stdout.strip()

        source_docs = checkout / KUBERNETES_DOCS_SUBPATH
        if not source_docs.exists():
            raise FileNotFoundError(
                f"Kubernetes docs path missing after checkout: {source_docs}"
            )
        shutil.copytree(source_docs, output_path / KUBERNETES_DOCS_SUBPATH)

    fetched_at = now or datetime.now(UTC)
    manifest = {
        "repo_url": repo_url,
        "requested_ref": ref,
        "resolved_sha": resolved_sha,
        "docs_path": str(KUBERNETES_DOCS_SUBPATH),
        "fetched_at": fetched_at.isoformat(),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _docs_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    nested = root / KUBERNETES_DOCS_SUBPATH
    if nested.exists():
        return nested
    return root


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            payload: dict[str, str] = {}
            for raw in lines[1:index]:
                if ":" not in raw:
                    continue
                key, value = raw.split(":", 1)
                payload[key.strip()] = value.strip().strip("\"'")
            return payload, "\n".join(lines[index + 1 :])
    return {}, text


def _strip_hugo_noise(text: str) -> str:
    text = re.sub(
        r"\{\{[%<]\s*([a-zA-Z0-9_-]+).*?[%>]\}\}.*?\{\{[%<]\s*/\1\s*[%>]\}\}",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"\{\{[%<].*?[%>]\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return text.strip()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "overview"


def _doc_id(path: Path, docs_root: Path) -> str:
    relative = path.relative_to(docs_root).with_suffix("")
    if relative.name == "index":
        relative = relative.parent
    return relative.as_posix() or "index"


def _section_records(
    *,
    body: str,
    doc_id: str,
    fallback_title: str,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_title = fallback_title
    current_slug = "overview"
    current_lines: list[str] = []
    position = 0

    def flush() -> None:
        nonlocal position, current_lines
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            return
        span_id = f"{doc_id}#{current_slug}"
        sections.append(
            {
                "id_sp": span_id,
                "tag": "section",
                "start_sp": position,
                "end_sp": position + len(text),
                "text_sp": text + "\n",
                "title": current_title,
                "parent_titles": [],
                "id_sec": current_slug,
                "start_sec": position,
                "end_sec": position + len(text),
                "text_sec": text,
            }
        )
        position += len(text) + 1
        current_lines = []

    for line in body.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush()
            current_title = heading.group(2).strip()
            current_slug = _slug(current_title)
            current_lines = [current_title]
            continue
        current_lines.append(line)
    flush()

    if sections:
        return sections
    return [
        {
            "id_sp": f"{doc_id}#overview",
            "tag": "section",
            "start_sp": 0,
            "end_sp": len(fallback_title),
            "text_sp": fallback_title + "\n",
            "title": fallback_title,
            "parent_titles": [],
            "id_sec": "overview",
            "start_sec": 0,
            "end_sec": len(fallback_title),
            "text_sec": fallback_title,
        }
    ]


def load_kubernetes_documents(dataset_root: str | Path) -> list[Document]:
    docs_root = _docs_root(dataset_root)
    if not docs_root.exists():
        raise FileNotFoundError(f"Kubernetes docs root not found: {docs_root}")

    documents: list[Document] = []
    for path in sorted(docs_root.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw)
        body = _strip_hugo_noise(body)
        doc_id = _doc_id(path, docs_root)
        title = (
            frontmatter.get("title")
            or doc_id.rsplit("/", 1)[-1].replace("-", " ").title()
        )
        spans = _section_records(body=body, doc_id=doc_id, fallback_title=title)
        documents.append(
            {
                "domain": "kubernetes",
                "doc_id": doc_id,
                "title": title,
                "doc_text": body,
                "doc_html_ts": json.dumps(frontmatter, ensure_ascii=True),
                "doc_html_raw": raw,
                "spans": spans,
                "raw_spans": {span["id_sp"]: span for span in spans},
            }
        )
    return documents
