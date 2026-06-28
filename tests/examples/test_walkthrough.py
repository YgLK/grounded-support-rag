from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from support_graph.config.runtime import ConfigValidationError

from examples import _shared


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_example_script(script_name: str):
    path = EXAMPLES_DIR / script_name
    spec = importlib.util.spec_from_file_location(
        f"example_{script_name.replace('.', '_').replace('-', '_')}", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_example_scripts_import_without_errors() -> None:
    script_paths = sorted(EXAMPLES_DIR.glob("[0-9][0-9]_*.py"))

    loaded = [_load_example_script(path.name) for path in script_paths]

    assert len(loaded) == 11
    assert all(hasattr(module, "build_lines") for module in loaded)


def test_shared_helper_exposes_repo_root_sample_ids_and_friendly_guidance() -> None:
    settings = _shared.load_settings()
    sample_ids = _shared.walkthrough_sample_ids()
    examples = _shared.load_or_build_examples(settings)

    assert _shared.repo_root() == REPO_ROOT
    assert sample_ids["example_id"] == "dmv::1409501a35697e0ce68561e29577b90a::turn_2"
    assert any(
        example["example_id"] == sample_ids["example_id"] for example in examples
    )

    lines = _shared.missing_config_lines("SupportGraph Example", ["FIELD_A", "FIELD_B"])
    assert lines[:4] == [
        "SupportGraph Example",
        "State: missing-config",
        "Missing config",
        "FIELD_A, FIELD_B",
    ]
    assert "Next" in lines


def test_examples_00_through_06_emit_expected_local_walkthrough_content() -> None:
    expected_substrings = {
        "00_project_overview.py": [
            "support_graph/runtime/graph.py",
            "build-chunks, build-examples, build-subsets",
            "embedding model",
        ],
        "01_dataset_eda.py": [
            "Document count: 488",
            "{'dmv': 149, 'ssa': 109, 'studentaid': 92, 'va': 138}",
            "Sample dialogue ID: 1409501a35697e0ce68561e29577b90a",
        ],
        "02_document_anatomy.py": [
            "Doc ID: Registrations#3_0",
            "Span count: 80",
            "Vehicles already registered in New York",
        ],
        "03_dialogue_and_example_eda.py": [
            "Example ID: dmv::1409501a35697e0ce68561e29577b90a::turn_2",
            "latest_user_utterance: My insurance ended so what should i do",
            "gold_span_ids: ['24', '25', '26']",
        ],
        "04_chunking_eda.py": [
            "Chunk count: 4316",
            "dmv::Registrations#3_0::sec::4::sub::0",
            "dmv::doc-1::sec::10::sub::1",
        ],
        "05_subset_eda.py": [
            "Matches committed file: yes",
            "Sample example membership",
            "Size: 200",
        ],
        "06_query_and_retrieval_eda.py": [
            "Latest user need: My insurance ended so what should i do",
            "chunk_id dmv::Top 5 DMV Mistakes and How to Avoid Them#3_0::sec::7::sub::0",
            "rerank_score",
        ],
    }

    for script_name, substrings in expected_substrings.items():
        module = _load_example_script(script_name)
        output = "\n".join(module.build_lines())
        for substring in substrings:
            assert substring in output


def test_examples_07_through_09_show_prerequisite_guidance_without_live_services() -> (
    None
):
    missing_index_settings = SimpleNamespace(
        runtime=SimpleNamespace(
            validate_for_index=lambda: (_ for _ in ()).throw(
                ConfigValidationError(
                    scope="index",
                    missing_fields=[
                        ".env: SUPPORT_GRAPH_POSTGRES_DSN",
                        "support_graph.toml: runtime.embedding_model",
                    ],
                )
            )
        )
    )
    missing_runtime_settings = SimpleNamespace(
        runtime=SimpleNamespace(
            validate_for_run=lambda: (_ for _ in ()).throw(
                ConfigValidationError(
                    scope="runtime",
                    missing_fields=[
                        ".env: SUPPORT_GRAPH_POSTGRES_DSN",
                        "support_graph.toml: runtime.chat_model",
                    ],
                )
            )
        )
    )

    module_07 = _load_example_script("07_index_and_vectorstore_walkthrough.py")
    module_08 = _load_example_script("08_run_graph_example.py")
    module_09 = _load_example_script("09_eval_smoke_walkthrough.py")

    output_07 = "\n".join(module_07.build_lines(settings=missing_index_settings))
    output_08 = "\n".join(module_08.build_lines(settings=missing_runtime_settings))
    output_09 = "\n".join(module_09.build_lines(settings=missing_runtime_settings))

    assert "State: missing-config" in output_07
    assert "uv run grounded-support-rag index-docs --domain dmv" in output_07
    assert "State: missing-config" in output_08
    assert "cp .env.example .env" in output_08
    assert "State: missing-config" in output_09
    assert "docker compose up -d postgres" in output_09


def test_example_10_shows_guidance_when_no_eval_artifacts_exist(tmp_path: Path) -> None:
    module = _load_example_script("10_trace_and_failure_review.py")
    settings = SimpleNamespace(
        paths=SimpleNamespace(eval_runs_dir=tmp_path / "outputs/evals/runs")
    )

    output = "\n".join(module.build_lines(settings=settings))

    assert "State: artifacts-missing" in output
    assert "uv run python examples/09_eval_smoke_walkthrough.py" in output
    assert "uv run grounded-support-rag review-failures --run-id <run_id>" in output
