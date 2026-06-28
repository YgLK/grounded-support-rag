from pathlib import Path

import pytest

from support_graph.data.dataset import load_dialogues, load_documents


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = REPO_ROOT / "multidoc2dial"


def test_load_documents_filters_to_dmv() -> None:
    documents = load_documents(DATASET_ROOT, domains=["dmv"])

    assert len(documents) == 149
    assert {document["domain"] for document in documents} == {"dmv"}
    assert documents[0]["doc_id"] == "A Guide for Facilities:#3_0"
    assert (
        documents[-1]["doc_id"]
        == "Voter Registration Application Frequently Asked Questions#1_0"
    )


def test_load_documents_normalizes_spans_in_order() -> None:
    documents = load_documents(DATASET_ROOT, domains=["dmv"])
    registrations = next(
        document for document in documents if document["doc_id"] == "Registrations#3_0"
    )

    assert registrations["spans"][0]["id_sp"] == "1"
    assert registrations["spans"][0]["start_sp"] == 0
    assert (
        registrations["spans"][5]["title"] == "Vehicles already registered in New York"
    )


def test_load_dialogues_filters_domain_and_sorts_turns() -> None:
    dialogues = load_dialogues(DATASET_ROOT, split="validation", domains=["dmv"])

    assert len(dialogues) == 165
    assert {dialogue["domain"] for dialogue in dialogues} == {"dmv"}
    assert dialogues[0]["dial_id"]
    assert dialogues[0]["turns"][0]["turn_id"] == 1


def test_load_dialogues_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        load_dialogues(DATASET_ROOT, split="dev", domains=["dmv"])
