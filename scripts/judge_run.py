"""Script to run LLM-as-a-judge on an existing eval run."""

import asyncio
import json
import argparse
from pathlib import Path
from support_graph.config.settings import Settings
from support_graph.evaluation.judge import RAGJudge
from support_graph.logging_utils import get_logger

logger = get_logger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge on a run's predictions."
    )
    parser.add_argument("run_dir", type=str, help="Path to the eval run directory")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use as judge (defaults to config)",
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of examples to judge"
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    predictions_path = run_path / "predictions.jsonl"
    if not predictions_path.exists():
        print(f"Predictions file not found: {predictions_path}")
        return

    settings = Settings.load()
    # Use the config as-is unless a model override is provided
    config = settings.runtime
    if args.model:
        config = config.with_overrides(chat_model=args.model)

    judge = RAGJudge.from_config(config)

    print(f"--- Judging run: {run_path.name} ---")
    print(f"Using judge model: {config.chat_model} ({config.chat_provider_type})")

    with open(predictions_path, "r") as f:
        lines = f.readlines()

    if args.limit:
        lines = lines[: args.limit]

    results = []
    for line in lines:
        pred = json.loads(line)
        query = pred.get("latest_user_utterance", "")
        answer = pred.get("response_text", "")
        # Combine retrieved chunks into a single context string
        chunks = pred.get("retrieved_chunks", [])
        if not chunks:
            chunks = pred.get("retrieval_ranked_chunks", [])

        context = "\n\n".join([c.get("text", "") for c in chunks if c.get("text")])

        if not context.strip():
            print(f"Warning: Empty context for example {pred.get('example_id')}")

        if pred.get("decision") != "answer":
            print(
                f"Skipping example {pred.get('example_id')} (Decision: {pred.get('decision')})"
            )
            continue

        print(f"Judging example: {pred.get('example_id')}...")
        triad = await judge.evaluate_triad(query, context, answer)

        print(
            f"  > Faithfulness: {triad['faithfulness'].score:.2f} ({triad['faithfulness'].reason})"
        )
        print(
            f"  > Answer Relevance: {triad['answer_relevance'].score:.2f} ({triad['answer_relevance'].reason})"
        )
        print(
            f"  > Context Relevance: {triad['context_relevance'].score:.2f} ({triad['context_relevance'].reason})"
        )

        results.append(
            {
                "example_id": pred.get("example_id"),
                "triad": {
                    k: {"score": v.score, "reason": v.reason} for k, v in triad.items()
                },
            }
        )

    # Save judge results
    output_path = run_path / "judge_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Judge results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
