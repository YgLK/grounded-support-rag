"""Dependency injection utilities for FastAPI routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.templating import Jinja2Templates

if TYPE_CHECKING:
    from support_graph.config.settings import Settings
    from support_graph.ui.loaders import WorkbenchArtifactLoader


_loader: WorkbenchArtifactLoader | None = None
_templates: Jinja2Templates | None = None
_settings: Settings | None = None


def init_dependencies(
    loader: WorkbenchArtifactLoader,
    templates: Jinja2Templates,
    settings: Settings,
) -> None:
    global _loader, _templates, _settings
    _loader = loader
    _templates = templates
    _settings = settings


def get_loader() -> WorkbenchArtifactLoader:
    if _loader is None:
        raise RuntimeError("Loader not initialized")
    return _loader


def get_templates() -> Jinja2Templates:
    if _templates is None:
        raise RuntimeError("Templates not initialized")
    return _templates


def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("Settings not initialized")
    return _settings
