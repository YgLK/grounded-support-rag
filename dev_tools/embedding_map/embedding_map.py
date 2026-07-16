"""Embedding-space export, projection, plotting, and diagnostics."""

from __future__ import annotations

import argparse
import colorsys
import csv
import html
import json
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psycopg
from sklearn.decomposition import PCA

from support_graph.config.settings import Settings
from support_graph.evaluation.evaluate import _prediction_retrieval_ranked_chunks
from support_graph.retrieval.index import psycopg_connection_string


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    token_count: int | None
    path_prefix: str
    text: str
    embedding: tuple[float, ...]
    overlay_status: str = "corpus"


@dataclass(frozen=True, slots=True)
class ProjectionPoint:
    chunk_id: str
    x: float
    y: float


def parse_embedding(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        if not stripped:
            return ()
        return tuple(float(part.strip()) for part in stripped.split(","))
    if isinstance(value, np.ndarray):
        return tuple(float(part) for part in value.tolist())
    if isinstance(value, Sequence):
        return tuple(float(part) for part in value)
    raise TypeError(f"Unsupported embedding value: {type(value).__name__}")


def path_prefix(doc_id: str, depth: int = 2) -> str:
    parts = [part for part in doc_id.split("/") if part]
    return "/".join(parts[:depth]) if parts else "(unknown)"


def _metadata_value(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    return str(value) if value is not None else ""


def parse_pgvector_row(row: Mapping[str, Any]) -> EmbeddingRecord:
    metadata = row.get("metadata") or row.get("cmetadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, Mapping):
        raise TypeError("pgvector metadata must be a mapping or JSON object string.")

    doc_id = _metadata_value(metadata, "doc_id")
    token_count_raw = metadata.get("token_count")
    token_count = int(token_count_raw) if token_count_raw is not None else None
    text = str(row.get("text") or row.get("document") or "")
    return EmbeddingRecord(
        chunk_id=_metadata_value(metadata, "chunk_id"),
        doc_id=doc_id,
        doc_title=_metadata_value(metadata, "doc_title"),
        section_title=_metadata_value(metadata, "section_title"),
        token_count=token_count,
        path_prefix=path_prefix(doc_id),
        text=text,
        embedding=parse_embedding(row["embedding"]),
    )


def _sample_evenly(
    records: list[EmbeddingRecord], sample_size: int | None
) -> list[EmbeddingRecord]:
    if sample_size is None or sample_size >= len(records):
        return records
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    step = len(records) / sample_size
    return [
        records[min(len(records) - 1, math.floor(index * step))]
        for index in range(sample_size)
    ]


def fetch_pgvector_records(
    *,
    postgres_dsn: str,
    collection_name: str,
    doc_prefixes: Sequence[str] = (),
    sample_size: int | None = None,
) -> list[EmbeddingRecord]:
    query = """
        select e.embedding::text as embedding, e.document as text, e.cmetadata as metadata
        from langchain_pg_embedding e
        join langchain_pg_collection c on e.collection_id = c.uuid
        where c.name = %s
        order by e.cmetadata->>'chunk_id'
    """
    records: list[EmbeddingRecord] = []
    with psycopg.connect(
        psycopg_connection_string(postgres_dsn),
        connect_timeout=5,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (collection_name,))
            for embedding, text, metadata in cur.fetchall():
                record = parse_pgvector_row(
                    {"embedding": embedding, "text": text, "metadata": metadata}
                )
                if doc_prefixes and not any(
                    record.doc_id.startswith(prefix) for prefix in doc_prefixes
                ):
                    continue
                records.append(record)
    return _sample_evenly(records, sample_size)


def record_to_json(record: EmbeddingRecord) -> dict[str, Any]:
    return {
        "chunk_id": record.chunk_id,
        "doc_id": record.doc_id,
        "doc_title": record.doc_title,
        "section_title": record.section_title,
        "token_count": record.token_count,
        "path_prefix": record.path_prefix,
        "text": record.text,
        "embedding": list(record.embedding),
        "overlay_status": record.overlay_status,
    }


def write_embeddings_jsonl(records: Sequence[EmbeddingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_json(record), ensure_ascii=True))
            handle.write("\n")


def write_plot_cache(records: Sequence[EmbeddingRecord], path: Path) -> None:
    """Write a lightweight cache with everything the plot needs except the
    embedding vectors, so frontend tweaks can re-render without touching the
    large embeddings.jsonl artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record_to_json(record)
            payload.pop("embedding", None)
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")


def load_embeddings_jsonl(path: Path) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        embedding = payload.get("embedding") or ()
        records.append(
            EmbeddingRecord(
                chunk_id=str(payload["chunk_id"]),
                doc_id=str(payload["doc_id"]),
                doc_title=str(payload.get("doc_title", "")),
                section_title=str(payload.get("section_title", "")),
                token_count=payload.get("token_count"),
                path_prefix=str(
                    payload.get("path_prefix") or path_prefix(str(payload["doc_id"]))
                ),
                text=str(payload.get("text", "")),
                embedding=tuple(float(part) for part in embedding),
                overlay_status=str(payload.get("overlay_status", "corpus")),
            )
        )
    return records


def _embedding_matrix(records: Sequence[EmbeddingRecord]) -> np.ndarray:
    if not records:
        raise ValueError("No embedding records available.")
    return np.array([record.embedding for record in records], dtype=float)


def project_pca(records: Sequence[EmbeddingRecord]) -> list[ProjectionPoint]:
    matrix = _embedding_matrix(records)
    if len(records) == 1:
        coords = np.array([[0.0, 0.0]])
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
    return [
        ProjectionPoint(chunk_id=record.chunk_id, x=float(x), y=float(y))
        for record, (x, y) in zip(records, coords, strict=True)
    ]


def project_umap(records: Sequence[EmbeddingRecord]) -> list[ProjectionPoint]:
    matrix = _embedding_matrix(records)
    if len(records) == 1:
        coords = np.array([[0.0, 0.0]])
    else:
        import umap

        n_neighbors = min(15, max(2, len(records) - 1))
        coords = umap.UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
        ).fit_transform(matrix)
    return [
        ProjectionPoint(chunk_id=record.chunk_id, x=float(x), y=float(y))
        for record, (x, y) in zip(records, coords, strict=True)
    ]


def write_projection_csv(points: Sequence[ProjectionPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chunk_id", "x", "y"])
        writer.writeheader()
        for point in points:
            writer.writerow({"chunk_id": point.chunk_id, "x": point.x, "y": point.y})


def load_projection_csv(path: Path) -> list[ProjectionPoint]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            ProjectionPoint(
                chunk_id=row["chunk_id"],
                x=float(row["x"]),
                y=float(row["y"]),
            )
            for row in csv.DictReader(handle)
        ]


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 1.0
    return 1.0 - float(np.dot(left, right) / denom)


def nearest_neighbors(
    records: Sequence[EmbeddingRecord],
    *,
    limit: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    matrix = _embedding_matrix(records)
    result: dict[str, list[tuple[str, float]]] = {}
    for index, record in enumerate(records):
        distances = [
            (other.chunk_id, _cosine_distance(matrix[index], matrix[other_index]))
            for other_index, other in enumerate(records)
            if other_index != index
        ]
        result[record.chunk_id] = sorted(distances, key=lambda item: item[1])[:limit]
    return result


def path_prefix_purity(
    records: Sequence[EmbeddingRecord],
    *,
    neighbor_count: int = 5,
) -> float | None:
    if len(records) <= 1:
        return None
    neighbors = nearest_neighbors(records, limit=min(neighbor_count, len(records) - 1))
    by_id = {record.chunk_id: record for record in records}
    scores: list[float] = []
    for record in records:
        near = neighbors[record.chunk_id]
        if not near:
            continue
        matched = sum(
            1
            for chunk_id, _ in near
            if by_id[chunk_id].path_prefix == record.path_prefix
        )
        scores.append(matched / len(near))
    return sum(scores) / len(scores) if scores else None


def suspicious_neighbors(
    records: Sequence[EmbeddingRecord],
    *,
    limit: int = 10,
) -> list[tuple[EmbeddingRecord, EmbeddingRecord, float]]:
    neighbors = nearest_neighbors(records, limit=1)
    by_id = {record.chunk_id: record for record in records}
    pairs: list[tuple[EmbeddingRecord, EmbeddingRecord, float]] = []
    for record in records:
        near = neighbors.get(record.chunk_id, [])
        if not near:
            continue
        neighbor_id, distance = near[0]
        neighbor = by_id[neighbor_id]
        if record.path_prefix != neighbor.path_prefix:
            pairs.append((record, neighbor, distance))
    return sorted(pairs, key=lambda item: item[2])[:limit]


def _status_rank(status: str) -> int:
    ranks = {
        "expected": 5,
        "acceptable": 4,
        "retrieved": 3,
        "citation": 2,
        "corpus": 1,
    }
    return ranks.get(status, 0)


def _merge_status(existing: str, candidate: str) -> str:
    return candidate if _status_rank(candidate) > _status_rank(existing) else existing


def load_eval_overlay(run_dir: Path) -> dict[str, str]:
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Eval predictions artifact not found: {predictions_path}"
        )
    status_by_chunk: dict[str, str] = {}
    status_by_doc: dict[str, str] = {}
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for doc_id in record.get("acceptable_sources", []):
            status_by_doc[str(doc_id)] = _merge_status(
                status_by_doc.get(str(doc_id), "corpus"), "acceptable"
            )
        for doc_id in record.get("expected_sources", []):
            status_by_doc[str(doc_id)] = _merge_status(
                status_by_doc.get(str(doc_id), "corpus"), "expected"
            )
        for chunk in _prediction_retrieval_ranked_chunks(record):
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id:
                status_by_chunk[chunk_id] = _merge_status(
                    status_by_chunk.get(chunk_id, "corpus"), "retrieved"
                )
        for citation in record.get("citations", []):
            chunk_id = str(citation.get("chunk_id", ""))
            if chunk_id:
                status_by_chunk[chunk_id] = _merge_status(
                    status_by_chunk.get(chunk_id, "corpus"), "citation"
                )
    return {
        **{f"doc:{key}": value for key, value in status_by_doc.items()},
        **status_by_chunk,
    }


def apply_overlay_status(
    records: Sequence[EmbeddingRecord], overlay: Mapping[str, str]
) -> list[EmbeddingRecord]:
    updated: list[EmbeddingRecord] = []
    for record in records:
        status = (
            overlay.get(record.chunk_id)
            or overlay.get(f"doc:{record.doc_id}")
            or "corpus"
        )
        updated.append(
            EmbeddingRecord(
                chunk_id=record.chunk_id,
                doc_id=record.doc_id,
                doc_title=record.doc_title,
                section_title=record.section_title,
                token_count=record.token_count,
                path_prefix=record.path_prefix,
                text=record.text,
                embedding=record.embedding,
                overlay_status=status,
            )
        )
    return updated


def _hover_field(label: str, value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"<b>{html.escape(label)}:</b> {html.escape(str(value))}<br>"


def _hover_text(text: str, *, width: int = 72, max_chars: int = 360) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    truncated = len(collapsed) > max_chars
    snippet = collapsed[:max_chars].rstrip()
    lines = textwrap.wrap(snippet, width=width) or [snippet]
    body = "<br>".join(html.escape(line) for line in lines)
    if truncated:
        body += "&#8230;"
    return f"<br><b>text:</b><br>{body}"


def _distinct_colors(count: int) -> list[str]:
    """Generate `count` visually distinct hex colors via golden-ratio hue
    spacing, cycling saturation/value so adjacent hues stay separable on a
    white background. Handles far more categories than Plotly's default 10."""
    if count <= 0:
        return []
    sv_cycle = [(0.72, 0.80), (0.55, 0.62), (0.85, 0.68), (0.45, 0.88)]
    golden = 0.618033988749895
    colors: list[str] = []
    hue = 0.1
    for index in range(count):
        hue = (hue + golden) % 1.0
        saturation, value = sv_cycle[index % len(sv_cycle)]
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(
            f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
        )
    return colors


def _build_hover(record: EmbeddingRecord) -> str:
    header = html.escape(record.section_title or record.doc_title or record.doc_id)
    return (
        f"<b>{header}</b><br>"
        f"{_hover_field('doc_id', record.doc_id)}"
        f"{_hover_field('path_prefix', record.path_prefix)}"
        f"{_hover_field('chunk_id', record.chunk_id)}"
        f"{_hover_field('token_count', record.token_count)}"
        f"{_hover_field('overlay_status', record.overlay_status)}"
        f"{_hover_text(record.text)}"
    )


def write_plot_html(
    records: Sequence[EmbeddingRecord],
    points: Sequence[ProjectionPoint],
    path: Path,
    *,
    title: str,
    color_by: str = "path_prefix",
) -> None:
    import plotly.graph_objects as go

    point_by_chunk = {point.chunk_id: point for point in points}
    if color_by not in {"path_prefix", "doc_id", "overlay_status"}:
        raise ValueError("color_by must be one of: path_prefix, doc_id, overlay_status")

    groups: dict[str, list[EmbeddingRecord]] = {}
    for record in records:
        groups.setdefault(str(getattr(record, color_by)), []).append(record)

    fig = go.Figure()
    sorted_groups = sorted(groups.items())
    palette = _distinct_colors(len(sorted_groups))
    for color, (group, group_records) in zip(palette, sorted_groups):
        fig.add_trace(
            go.Scatter(
                x=[point_by_chunk[record.chunk_id].x for record in group_records],
                y=[point_by_chunk[record.chunk_id].y for record in group_records],
                mode="markers",
                name=group,
                legendgroup=group,
                marker={
                    "size": 7,
                    "opacity": 0.8,
                    "color": color,
                    "line": {"width": 0.5, "color": "rgba(40,40,40,0.35)"},
                },
                customdata=[_build_hover(record) for record in group_records],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 18}},
        template="plotly_white",
        xaxis_title="x",
        yaxis_title="y",
        hovermode="closest",
        hoverlabel={
            "align": "left",
            "bgcolor": "rgba(255,255,255,0.95)",
            "bordercolor": "#999",
            "font": {"size": 12, "family": "system-ui, sans-serif"},
        },
        legend={
            "title": {"text": color_by, "font": {"size": 12}},
            "itemsizing": "constant",
            "x": 1.01,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#ddd",
            "borderwidth": 1,
            "font": {"size": 10},
        },
        margin={"l": 60, "r": 260, "t": 70, "b": 60},
        autosize=True,
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height="100vh",
        config={"responsive": True, "scrollZoom": True},
    )


def write_summary(
    records: Sequence[EmbeddingRecord],
    path: Path,
    *,
    projection_names: Sequence[str],
) -> None:
    purity = path_prefix_purity(records)
    suspicious = suspicious_neighbors(records)
    prefix_counts: dict[str, int] = {}
    for record in records:
        prefix_counts[record.path_prefix] = prefix_counts.get(record.path_prefix, 0) + 1
    lines = [
        "# Embedding Map Summary",
        "",
        "UMAP and PCA plots are diagnostic views, not retrieval metrics. Judge retrieval quality with recall/MRR/NDCG, citations, required points, and failure labels.",
        "",
        f"- Records: {len(records)}",
        f"- Projections: {', '.join(projection_names)}",
        f"- Path-prefix purity@5: {purity:.3f}"
        if purity is not None
        else "- Path-prefix purity@5: n/a",
        "",
        "## Path Prefix Counts",
        "",
    ]
    for prefix, count in sorted(
        prefix_counts.items(), key=lambda item: (-item[1], item[0])
    )[:20]:
        lines.append(f"- `{prefix}`: {count}")
    lines.extend(["", "## Suspicious Nearest Neighbors", ""])
    if suspicious:
        for left, right, distance in suspicious:
            lines.append(
                f"- `{left.chunk_id}` near `{right.chunk_id}` distance={distance:.4f} prefixes `{left.path_prefix}` vs `{right.path_prefix}`"
            )
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir(project_root: Path, run_id: str | None = None) -> Path:
    resolved_run_id = run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return project_root / "outputs" / "dev_tools" / "embedding_map" / resolved_run_id


def run_export(args: argparse.Namespace) -> None:
    settings = Settings.load(args.config_file, args.secrets_file)
    domain = settings.selected_domain(args.domain)
    runtime = settings.runtime_for(domain)
    if runtime.postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for embedding export.")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(settings.paths.project_root, args.run_id)
    )
    records = fetch_pgvector_records(
        postgres_dsn=runtime.postgres_dsn,
        collection_name=settings.collection_name(domain),
        doc_prefixes=args.doc_prefix,
        sample_size=args.sample_size,
    )
    if args.eval_run_id:
        overlay = load_eval_overlay(settings.paths.eval_runs_dir / args.eval_run_id)
        records = apply_overlay_status(records, overlay)
    write_embeddings_jsonl(records, output_dir / "embeddings.jsonl")
    print(output_dir)


def run_project(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    records = load_embeddings_jsonl(output_dir / "embeddings.jsonl")
    if args.method in {"pca", "both"}:
        write_projection_csv(project_pca(records), output_dir / "projection-pca.csv")
    if args.method in {"umap", "both"}:
        write_projection_csv(project_umap(records), output_dir / "projection-umap.csv")


def run_plot(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    cache_path = output_dir / "plot-data.jsonl"
    refresh_cache = getattr(args, "refresh_cache", False)
    if cache_path.exists() and not refresh_cache:
        records = load_embeddings_jsonl(cache_path)
    else:
        records = load_embeddings_jsonl(output_dir / "embeddings.jsonl")
        write_plot_cache(records, cache_path)
    has_embeddings = bool(records and records[0].embedding)
    projections: list[str] = []
    pca_path = output_dir / "projection-pca.csv"
    if pca_path.exists():
        write_plot_html(
            records,
            load_projection_csv(pca_path),
            output_dir / "embedding-map-pca.html",
            title="SupportGraph Embedding Map - PCA",
            color_by=args.color_by,
        )
        projections.append("pca")
    umap_path = output_dir / "projection-umap.csv"
    if umap_path.exists():
        write_plot_html(
            records,
            load_projection_csv(umap_path),
            output_dir / "embedding-map-umap.html",
            title="SupportGraph Embedding Map - UMAP",
            color_by=args.color_by,
        )
        projections.append("umap")
    if has_embeddings:
        write_summary(records, output_dir / "summary.md", projection_names=projections)
    else:
        print(
            "Rendered from plot-data.jsonl cache; skipped summary "
            "(rerun with --refresh-cache to recompute diagnostics)."
        )


def run_all(args: argparse.Namespace) -> None:
    if not args.output_dir:
        settings = Settings.load(args.config_file, args.secrets_file)
        args.output_dir = str(
            default_output_dir(settings.paths.project_root, args.run_id)
        )
    run_export(args)
    output_dir = Path(args.output_dir)
    project_args = argparse.Namespace(output_dir=output_dir, method="both")
    run_project(project_args)
    plot_args = argparse.Namespace(output_dir=output_dir, color_by=args.color_by)
    run_plot(plot_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SupportGraph embedding map diagnostics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_export_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--config-file", default="support_graph.kubernetes.toml")
        target.add_argument("--secrets-file", default=None)
        target.add_argument("--domain", default="kubernetes")
        target.add_argument("--sample-size", type=int, default=None)
        target.add_argument("--doc-prefix", action="append", default=[])
        target.add_argument("--eval-run-id", default=None)
        target.add_argument("--run-id", default=None)
        target.add_argument("--output-dir", default=None)

    export_parser = subparsers.add_parser("export", help="Export pgvector embeddings.")
    add_export_args(export_parser)
    export_parser.set_defaults(func=run_export)

    project_parser = subparsers.add_parser(
        "project", help="Compute PCA/UMAP projections."
    )
    project_parser.add_argument("--output-dir", required=True)
    project_parser.add_argument(
        "--method", choices=["pca", "umap", "both"], default="both"
    )
    project_parser.set_defaults(func=run_project)

    plot_parser = subparsers.add_parser(
        "plot", help="Write Plotly HTML maps and summary."
    )
    plot_parser.add_argument("--output-dir", required=True)
    plot_parser.add_argument(
        "--color-by",
        choices=["path_prefix", "doc_id", "overlay_status"],
        default="path_prefix",
    )
    plot_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild plot-data.jsonl from embeddings.jsonl and recompute summary.",
    )
    plot_parser.set_defaults(func=run_plot)

    all_parser = subparsers.add_parser(
        "run", help="Export, project, plot, and summarize."
    )
    add_export_args(all_parser)
    all_parser.add_argument(
        "--color-by",
        choices=["path_prefix", "doc_id", "overlay_status"],
        default="path_prefix",
    )
    all_parser.set_defaults(func=run_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
