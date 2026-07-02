from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from support_graph.data.documents import load_documents
from support_graph.data.kubernetes import (
    KUBERNETES_DOCS_SUBPATH,
    fetch_kubernetes_docs,
    load_kubernetes_documents,
)


def _write_doc(root: Path, relative: str, text: str) -> Path:
    path = root / KUBERNETES_DOCS_SUBPATH / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_load_kubernetes_documents_parses_markdown_sections() -> None:
    root = Path("unused")
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _write_doc(
            root,
            "concepts/workloads/pods/index.md",
            """---
title: Pods
description: Pod docs
---
{{< note >}}noise{{< /note >}}

Kubernetes pod intro.

## What is a Pod?

A Pod is the smallest deployable compute object in Kubernetes.
""",
        )

        documents = load_kubernetes_documents(root)

    assert len(documents) == 1
    document = documents[0]
    assert document["domain"] == "kubernetes"
    assert document["doc_id"] == "concepts/workloads/pods"
    assert document["title"] == "Pods"
    assert [span["id_sp"] for span in document["spans"]] == [
        "concepts/workloads/pods#overview",
        "concepts/workloads/pods#what-is-a-pod",
    ]
    assert "noise" not in document["doc_text"]
    assert "smallest deployable" in document["spans"][1]["text_sp"]


def test_load_kubernetes_documents_normalizes_hugo_index_doc_ids(
    tmp_path: Path,
) -> None:
    _write_doc(
        tmp_path,
        "concepts/workloads/pods/_index.md",
        "---\ntitle: Pods\n---\nPod docs.\n",
    )
    _write_doc(
        tmp_path,
        "concepts/services-networking/service/index.md",
        "---\ntitle: Services\n---\nService docs.\n",
    )

    documents = load_kubernetes_documents(tmp_path)

    assert [document["doc_id"] for document in documents] == [
        "concepts/services-networking/service",
        "concepts/workloads/pods",
    ]
    assert [document["spans"][0]["id_sp"] for document in documents] == [
        "concepts/services-networking/service#overview",
        "concepts/workloads/pods#overview",
    ]


def test_load_documents_dispatches_kubernetes_domain(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        "tasks/run-application/run-stateless-application-deployment.md",
        "---\ntitle: Deployments\n---\nDeployment docs.\n",
    )

    documents = load_documents(tmp_path, domains=["kubernetes"])

    assert len(documents) == 1
    assert documents[0]["domain"] == "kubernetes"
    assert (
        documents[0]["doc_id"]
        == "tasks/run-application/run-stateless-application-deployment"
    )


def test_fetch_kubernetes_docs_uses_sparse_checkout_and_writes_manifest(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command, *, cwd, check, text, capture_output):
        del check, text, capture_output
        args = tuple(command[1:])
        calls.append(args)
        if args[:1] == ("checkout",):
            _write_doc(
                Path(cwd),
                "concepts/workloads/pods/index.md",
                "---\ntitle: Pods\n---\nPod docs.\n",
            )
        if args == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    output_dir = tmp_path / "raw/kubernetes/current"
    manifest = fetch_kubernetes_docs(
        output_dir,
        ref="v1.30.0",
        runner=fake_runner,
        now=datetime(2026, 6, 27, tzinfo=UTC),
    )

    assert ("sparse-checkout", "set", "content/en/docs") in calls
    assert ("fetch", "--depth", "1", "origin", "v1.30.0") in calls
    assert manifest["resolved_sha"] == "abc123"
    assert (output_dir / "manifest.json").exists()
    assert (
        output_dir / KUBERNETES_DOCS_SUBPATH / "concepts/workloads/pods/index.md"
    ).exists()


def test_fetch_kubernetes_docs_refuses_existing_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "raw/kubernetes/current"
    output_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        fetch_kubernetes_docs(
            output_dir,
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
        )
