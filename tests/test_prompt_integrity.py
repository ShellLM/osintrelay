"""
Tests for prompt file integrity.

Covers:
- Prompt files exist on disk
- Prompt files are non-empty and above a minimum length
- Prompt files contain their required format placeholders
- format() substitution with required kwargs does not raise KeyError
- System prompt contains a SECURITY section
- Image prompt mentions injection warning
"""

import re
from pathlib import Path

import pytest

# The prompts directory lives at socialosintagent/prompts/ relative to the
# project root. conftest.py's change_test_dir fixture sets cwd to the project
# root before any tests run, so we resolve from there rather than from __file__
# to avoid the path doubling up when tests live inside the socialosintagent package.
PROMPTS_DIR = Path("socialosintagent") / "prompts"


def _load_raw(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


PROMPT_SPECS = [
    ("system_analysis.prompt", ["{current_timestamp}"]),
    ("image_analysis.prompt", ["{context}"]),
]


@pytest.mark.parametrize("filename,_placeholders", PROMPT_SPECS)
def test_prompt_file_exists(filename, _placeholders):
    """Each prompt file must exist on disk."""
    assert (PROMPTS_DIR / filename).exists(), (
        f"Prompt file not found: {PROMPTS_DIR / filename}"
    )


@pytest.mark.parametrize("filename,_placeholders", PROMPT_SPECS)
def test_prompt_file_is_non_empty(filename, _placeholders):
    """Each prompt file must contain substantive content (> 100 chars)."""
    content = _load_raw(filename)
    assert len(content) > 100, (
        f"{filename} is suspiciously short ({len(content)} chars)"
    )


@pytest.mark.parametrize("filename,placeholders", PROMPT_SPECS)
def test_prompt_contains_required_placeholders(filename, placeholders):
    """Each prompt must contain its documented format placeholders."""
    content = _load_raw(filename)
    for placeholder in placeholders:
        assert placeholder in content, (
            f"Expected placeholder '{placeholder}' not found in {filename}"
        )


@pytest.mark.parametrize("filename,placeholders", PROMPT_SPECS)
def test_prompt_format_substitution_succeeds(filename, placeholders):
    """format() with required kwargs must not raise KeyError.

    If this fails it means someone added a bare {placeholder} to the prompt
    file without registering it here — both places need updating together.
    """
    content = _load_raw(filename)
    kwargs = {p.strip("{}") : "TEST_VALUE" for p in placeholders}
    try:
        result = content.format(**kwargs)
    except KeyError as exc:
        pytest.fail(
            f"{filename} has an unregistered placeholder that raises KeyError: {exc}. "
            "Register it in PROMPT_SPECS in this test file."
        )
    assert "TEST_VALUE" in result


def test_system_prompt_contains_security_section():
    """System prompt must include a SECURITY section to guard against injection."""
    content = _load_raw("system_analysis.prompt")
    assert "SECURITY" in content.upper()


def test_image_prompt_mentions_injection_warning():
    """Image prompt must warn the vision model about visual prompt injection."""
    content = _load_raw("image_analysis.prompt")
    assert "injection" in content.lower()
