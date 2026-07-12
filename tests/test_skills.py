"""Skill-freshness gate: `.claude/skills/` must describe THIS repo.

The DevOps skills are operating knowledge for humans and LLMs; stale
operating knowledge is worse than none. This suite machine-checks every
fact class a skill can pin — file paths, version pins, CLI surface, error
taxonomy, pytest themes, the CLAUDE.md registry — so that a major change
landing without its skill update turns CI red (the wiring promised by
CLAUDE.md's maintenance contract). When one of these tests fails, update
the SKILL, never weaken the test.

Runs as ``framework`` (every invocation, every CI build).
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.framework

ROOT = Path(__file__).parents[1]
SKILLS_DIR = ROOT / ".claude" / "skills"
CLAUDE_MD = ROOT / "CLAUDE.md"

EXPECTED_SKILLS = {
    "developing", "testing", "ci-cd", "debugging", "live-ops", "environment",
}

# Repo prefixes whose backticked mentions must exist in the checkout.
# runtime/, models/, data/ are operational or gitignored — excluded.
_CHECKED_PREFIXES = ("src/", "docs/", "scripts/", "tests/", ".github/", ".claude/")
_CHECKED_ROOT_FILES = {
    "requirements.txt", "pytest.ini", "ruff.toml",
    "CHANGELOG.md", "README.md", "CLAUDE.md", ".gitignore",
}
_PATH_TOKEN = re.compile(r"^[\w./\-]+$")


def _skill_dirs() -> list[Path]:
    return sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())


def _skill_text(name: str) -> str:
    return (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"\A---\n(.*?)\n---\n", text, flags=re.DOTALL)
    assert m, "SKILL.md must start with a --- frontmatter block"
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        if _:
            fields[k.strip()] = v.strip()
    return fields


def _backtick_spans(text: str) -> list[str]:
    return re.findall(r"`([^`\n]+)`", text)


# ---------------------------------------------------------------------------
# Structure: the folder holds exactly the registered skills, nothing stale
# ---------------------------------------------------------------------------


def test_skills_folder_contains_exactly_the_expected_skills():
    assert SKILLS_DIR.is_dir(), "skills folder missing"
    names = {p.name for p in _skill_dirs()}
    assert names == EXPECTED_SKILLS, (
        f"skills folder drifted: extra={names - EXPECTED_SKILLS}, "
        f"missing={EXPECTED_SKILLS - names}. New skill? Register it in "
        f"CLAUDE.md and in this test's EXPECTED_SKILLS."
    )
    strays = [p.name for p in SKILLS_DIR.iterdir() if not p.is_dir()]
    assert not strays, f"loose files in .claude/skills/: {strays}"


@pytest.mark.parametrize("name", sorted(EXPECTED_SKILLS))
def test_skill_dir_is_clean(name):
    """Only SKILL.md plus optional references/scripts/assets subdirs —
    anything else is stale clutter the maintenance contract forbids."""
    allowed_dirs = {"references", "scripts", "assets"}
    for entry in (SKILLS_DIR / name).iterdir():
        if entry.is_dir():
            assert entry.name in allowed_dirs, f"unexpected dir {entry}"
        else:
            assert entry.name == "SKILL.md", f"unexpected file {entry}"
    assert (SKILLS_DIR / name / "SKILL.md").is_file()


@pytest.mark.parametrize("name", sorted(EXPECTED_SKILLS))
def test_skill_frontmatter_and_update_triggers(name):
    text = _skill_text(name)
    fm = _frontmatter(text)
    assert fm.get("name") == name, "frontmatter name must equal the dir name"
    assert len(fm.get("description", "")) > 80, (
        "description is the triggering surface — keep it substantive"
    )
    assert "## Update triggers" in text, (
        "every skill carries its own maintenance contract section"
    )


# ---------------------------------------------------------------------------
# Path anchors: files a skill points at must exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_SKILLS))
def test_skill_referenced_paths_exist(name):
    missing = []
    for span in _backtick_spans(_skill_text(name)):
        token = span.split("::", 1)[0].rstrip("/")
        if not _PATH_TOKEN.match(token):
            continue  # commands, placeholders, expressions
        if token.startswith(_CHECKED_PREFIXES) or token in _CHECKED_ROOT_FILES:
            if not (ROOT / token).exists():
                missing.append(token)
    assert not missing, f"{name} skill references nonexistent paths: {missing}"


def test_claude_md_referenced_paths_exist():
    missing = []
    for span in _backtick_spans(CLAUDE_MD.read_text(encoding="utf-8")):
        token = span.split("::", 1)[0].rstrip("/")
        if not _PATH_TOKEN.match(token):
            continue
        if token.startswith(_CHECKED_PREFIXES) or token in _CHECKED_ROOT_FILES:
            if not (ROOT / token).exists():
                missing.append(token)
    assert not missing, f"CLAUDE.md references nonexistent paths: {missing}"


# ---------------------------------------------------------------------------
# Registry wiring: CLAUDE.md and the skills folder agree both ways
# ---------------------------------------------------------------------------


def test_claude_md_registry_matches_skill_dirs():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    for name in EXPECTED_SKILLS:
        assert f"`{name}`" in text, f"CLAUDE.md registry missing skill {name!r}"
    # Registry table rows: | `name` | ... — every row must be a real skill.
    listed = set(re.findall(r"^\|\s*`([a-z\-]+)`\s*\|", text, flags=re.MULTILINE))
    assert listed == EXPECTED_SKILLS, (
        f"CLAUDE.md registry rows {listed} != skill dirs {EXPECTED_SKILLS}"
    )


# ---------------------------------------------------------------------------
# Fact freshness per skill
# ---------------------------------------------------------------------------


def _requirement_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_environment_skill_pins_match_reality():
    from src.engine.environment import VALIDATED_STACK

    text = _skill_text("environment")
    pins = _requirement_pins()
    # Every pkg==ver the skill quotes must be the requirements.txt truth.
    quoted = re.findall(r"([A-Za-z0-9_.\-]+)==([0-9][A-Za-z0-9_.]*)", text)
    assert quoted, "environment skill must quote the pinned versions"
    for pkg, ver in quoted:
        assert pins.get(pkg.lower()) == ver, (
            f"environment skill quotes {pkg}=={ver} but requirements.txt "
            f"pins {pins.get(pkg.lower())} — update the skill"
        )
    # ...and it must quote the ENTIRE serving-critical manifest.
    for pkg, ver in VALIDATED_STACK.items():
        assert f"{pkg}=={ver}" in text, (
            f"environment skill missing serving-critical pin {pkg}=={ver}"
        )


def test_ci_skill_matches_workflow():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    text = _skill_text("ci-cd")
    ruff_pin = re.search(r"ruff==([\w.]+)", workflow)
    assert ruff_pin, "workflow no longer pins ruff — update this test + skill"
    assert f"ruff=={ruff_pin.group(1)}" in text, (
        f"ci-cd skill must quote the workflow's ruff pin {ruff_pin.group(0)}"
    )
    py = re.search(r'python-version:\s*"([\d.]+)"', workflow)
    assert py and py.group(1) in text, (
        f"ci-cd skill must mention the CI python version {py.group(1) if py else '?'}"
    )
    assert "requirements.txt" in text  # the cache/dependency key


def test_testing_skill_lists_every_theme():
    ini = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    m = re.search(r"^markers\s*=\s*\n(.*)\Z", ini, flags=re.MULTILINE | re.DOTALL)
    assert m, "pytest.ini markers section not found"
    themes = re.findall(r"^\s{4}(\w+):", m.group(1), flags=re.MULTILINE)
    assert len(themes) >= 10, f"theme extraction looks broken: {themes}"
    text = _skill_text("testing")
    missing = [t for t in themes if f"`{t}`" not in text]
    assert not missing, (
        f"testing skill's theme list is stale — missing {missing} "
        f"(pytest.ini is the source of truth)"
    )


def test_debugging_skill_covers_every_error_class():
    errors_src = (ROOT / "src" / "engine" / "errors.py").read_text(encoding="utf-8")
    classes = re.findall(r"^class (\w+)\(", errors_src, flags=re.MULTILINE)
    assert len(classes) >= 10, f"error extraction looks broken: {classes}"
    text = _skill_text("debugging")
    missing = [c for c in classes if c not in text]
    assert not missing, (
        f"debugging skill's error taxonomy is stale — missing {missing} "
        f"(src/engine/errors.py is the source of truth)"
    )


# ---------------------------------------------------------------------------
# Installability: the skills ship with the project
# ---------------------------------------------------------------------------


def test_skills_are_version_controlled():
    """Cloning must install the skills — nothing under .claude/skills/
    (or CLAUDE.md) may be gitignored."""
    paths = [str(CLAUDE_MD.relative_to(ROOT))] + [
        str((d / "SKILL.md").relative_to(ROOT)) for d in _skill_dirs()
    ]
    proc = subprocess.run(
        ["git", "check-ignore", *paths],
        cwd=ROOT, capture_output=True, text=True,
    )
    # exit 1 = nothing ignored; 0 = something matched an ignore rule.
    assert proc.returncode == 1, f"gitignored skill files: {proc.stdout}"


def test_package_skills_exports_and_installs(tmp_path):
    """scripts/package_skills.py is the install path for non-Claude-Code
    surfaces — one .skill zip per registered skill, and a directory-copy
    install mode. If this breaks, the skills stop shipping."""
    dist = tmp_path / "dist"
    proc = subprocess.run(
        [sys.executable, "scripts/package_skills.py", "--dist", str(dist)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for name in EXPECTED_SKILLS:
        bundle = dist / f"{name}.skill"
        assert bundle.is_file(), f"missing bundle {bundle.name}"
        assert f"{name}/SKILL.md" in zipfile.ZipFile(bundle).namelist()

    target = tmp_path / "agent-skills"
    proc = subprocess.run(
        [sys.executable, "scripts/package_skills.py",
         "--install", "--target", str(target)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for name in EXPECTED_SKILLS:
        assert (target / name / "SKILL.md").is_file()


def test_live_ops_skill_cli_surface_is_real():
    cli_src = (ROOT / "src" / "engine" / "__main__.py").read_text(encoding="utf-8")
    text = _skill_text("live-ops")
    # Completeness: every subcommand the CLI defines is named in the skill.
    commands = re.findall(r'add_parser\(\s*"([\w\-]+)"', cli_src)
    assert len(commands) >= 5, f"subcommand extraction looks broken: {commands}"
    missing = [c for c in commands if c not in text]
    assert not missing, f"live-ops skill missing CLI subcommands {missing}"
    # Accuracy: every flag the skill mentions actually exists in the CLI.
    real_flags = set(re.findall(r'"(--[\w\-]+)"', cli_src))
    fake = [f for f in set(re.findall(r"--[a-z][\w\-]+", text)) if f not in real_flags]
    assert not fake, (
        f"live-ops skill mentions flags the CLI does not define: {fake} "
        f"(src/engine/__main__.py is the source of truth)"
    )
