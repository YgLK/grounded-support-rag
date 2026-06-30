from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_readme_md_exists_for_package_metadata() -> None:
    """pyproject.toml declares readme = "README.md"; the file must exist.

    Deleting README.md breaks setuptools packaging/metadata flows because
    pyproject.toml still references it. This test guards against accidental
    deletion.
    """
    readme = REPO_ROOT / "README.md"
    assert readme.exists(), (
        "README.md is missing but pyproject.toml declares readme = 'README.md'. "
        "Restoring it is required for packaging/metadata."
    )
    assert readme.read_text(encoding="utf-8").strip(), (
        "README.md exists but is empty; pyproject.toml expects non-empty content."
    )
