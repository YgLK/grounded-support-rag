# ty Typecheck Cleanup Patterns

> Sources: SupportGraph local ty diagnostics and cleanup plan, 2026-07-02
> Raw: [ty Typecheck Cleanup Notes](../../raw/supportgraph/2026-07-02-ty-typecheck-cleanup-notes.md)

## Overview

`ty` found stale type contracts that Ruff could not see: mutable protocol attributes, `TypedDict` values passed to mutable `dict` parameters, partial test fixtures, and old call signatures. The durable lesson is to model read-only data as read-only in type signatures, fix shared contracts before call sites, and avoid suppressions until a real contract fix has been tried.

## Common Mistakes

- Mutable protocol members for config-like objects. Plain protocol attributes imply writable fields; frozen dataclasses such as `RuntimeConfig` do not satisfy that contract reliably. Fix by declaring config protocols with read-only `@property` members.
- `TypedDict` treated as plain `dict`. A `TypedDict` should not be accepted by a mutable `dict[...]` parameter unless the function truly needs destructive dict operations. Fix read-only consumers to accept `Mapping[str, Any]` or a narrower mapping protocol.
- `list[TypedDict]` passed to `list[dict]`. This combines mutable-dict mismatch with list invariance. Fix by using `Sequence[Mapping[str, Any]]` for readers, or exact domain lists such as `list[ChunkRecord]` when full shape matters.
- Tests using tiny partial dict literals for rich production shapes. Fix with typed fixture builders that provide default fields, or mark genuinely optional runtime keys as optional on the `TypedDict`.
- Stale APIs after refactors. Examples seen: removed `RuntimeConfig.with_overrides(...)` and a keyword-only `benchmark_embeddings(...)` call still being used positionally. Fix call sites to current APIs; do not paper over with casts.
- Helpers returning `object` across async boundaries. This hides concrete types and creates follow-on attribute errors. Fix with a `TypeVar`/generic return so sync and awaitable paths preserve the value type.
- Callback/event signatures typed as mutable `dict`. Graph stream events are `TypedDict`s; sinks that only inspect should accept `GraphStreamEvent` or read-only mappings.

## Fix Order

1. Run `uv run ty check --output-format concise` and count by rule/file.
2. Fix central protocols and shared type aliases first.
3. Change read-only APIs from `dict`/`list[dict]` to `Mapping`/`Sequence`.
4. Add test builders for repeated `TypedDict` fixtures.
5. Replace stale API calls with current helpers.
6. Re-run `uv run ty check`, then `uv run ruff check`, `uv run ruff format --check`, and targeted `uv run pytest`.

## Local Tooling Memory

Canonical ty workflow skill lives at `/Users/jakubszpunar/.codex/skills/ty/SKILL.md`. Use `uv run ty check` when `ty` is a project dependency; use `uvx ty` only for one-off projects without a dependency.

## See Also

- [Eval Expansion and Variance Attribution Tooling](eval-expansion-variance-tooling.md)
