"""Verify the configured OpenRouter chat and embedding models are reachable."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from support_graph.config.settings import Settings
from support_graph.providers import (
    Provider,
    build_chat_model,
    build_embeddings,
    chat_provider,
    embedding_provider,
)


DEFAULT_CHAT_PROMPT = "Reply with exactly OK."
DEFAULT_EMBED_TEXT = "support graph openrouter check"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ProviderCheckConfig:
    chat_provider_type: Provider
    embedding_provider_type: Provider
    ollama_base_url: str
    openrouter_base_url: str
    openrouter_api_key: str | None
    chat_model: str | None
    embedding_model: str | None
    embedding_client: Any | None = None


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(content).strip()


def _missing_openrouter_fields(
    settings: Settings,
    *,
    check_chat: bool,
    check_embeddings: bool,
) -> list[str]:
    missing: list[str] = []
    if check_chat and chat_provider(settings.runtime) is not Provider.OPENROUTER:
        missing.append('support_graph.toml: runtime.chat_provider_type = "openrouter"')
    if (
        check_embeddings
        and embedding_provider(settings.runtime) is not Provider.OPENROUTER
    ):
        missing.append(
            'support_graph.toml: runtime.embedding_provider_type = "openrouter"'
        )
    if (
        check_chat or check_embeddings
    ) and not settings.runtime.openrouter_api_key:
        missing.append(".env: SUPPORT_GRAPH_OPENROUTER_API_KEY")
    if check_chat and not settings.runtime.chat_model:
        missing.append("support_graph.toml: runtime.chat_model")
    if check_embeddings and not settings.runtime.embedding_model:
        missing.append("support_graph.toml: runtime.embedding_model")
    return missing


def _provider_config(settings: Settings) -> ProviderCheckConfig:
    return ProviderCheckConfig(
        chat_provider_type=chat_provider(settings.runtime),
        embedding_provider_type=embedding_provider(settings.runtime),
        ollama_base_url=settings.runtime.ollama_base_url,
        openrouter_base_url=settings.runtime.openrouter_base_url,
        openrouter_api_key=settings.runtime.openrouter_api_key,
        chat_model=settings.runtime.chat_model,
        embedding_model=settings.runtime.embedding_model,
    )


def run_checks(
    settings: Settings,
    *,
    check_chat: bool,
    check_embeddings: bool,
    chat_prompt: str = DEFAULT_CHAT_PROMPT,
    embed_text: str = DEFAULT_EMBED_TEXT,
) -> list[CheckResult]:
    missing = _missing_openrouter_fields(
        settings,
        check_chat=check_chat,
        check_embeddings=check_embeddings,
    )
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing or incompatible OpenRouter config: {joined}")

    provider_config = _provider_config(settings)
    results: list[CheckResult] = []

    if check_chat:
        model = build_chat_model(provider_config)
        response = model.invoke(chat_prompt)
        text = _extract_text(response)
        results.append(
            CheckResult(
                name="chat",
                ok=bool(text),
                detail=f"response={text or '<empty>'}",
            )
        )

    if check_embeddings:
        embeddings = build_embeddings(provider_config)
        vector = embeddings.embed_query(embed_text)
        results.append(
            CheckResult(
                name="embeddings",
                ok=bool(vector),
                detail=f"dimensions={len(vector)}",
            )
        )

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the configured OpenRouter chat and embedding models."
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional settings TOML path.",
    )
    parser.add_argument(
        "--secrets-file",
        default=None,
        help="Optional .env secrets path.",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip the chat model check.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the embeddings check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.load(
        Path(args.config_file) if args.config_file else None,
        Path(args.secrets_file) if args.secrets_file else None,
    )

    print("SupportGraph OpenRouter Check")
    print(f"config: {settings.paths.config_path}")
    print(f"secrets: {settings.paths.secrets_path}")
    print(f"chat_provider: {chat_provider(settings.runtime)}")
    print(f"embedding_provider: {embedding_provider(settings.runtime)}")
    print(f"base_url: {settings.runtime.openrouter_base_url}")
    print(f"chat_model: {settings.runtime.chat_model or '<missing>'}")
    print(f"embedding_model: {settings.runtime.embedding_model or '<missing>'}")

    try:
        results = run_checks(
            settings,
            check_chat=not args.skip_chat,
            check_embeddings=not args.skip_embeddings,
        )
    except Exception as exc:
        print("status: failed")
        print(f"error: {exc}")
        return 1

    if not results:
        print("status: skipped")
        print("error: both chat and embeddings checks were skipped")
        return 1

    for result in results:
        status = "ok" if result.ok else "failed"
        print(f"{result.name}: {status} ({result.detail})")

    if all(result.ok for result in results):
        print("status: ok")
        return 0

    print("status: failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
