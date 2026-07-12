"""Package or install the project's DevOps skills (.claude/skills/).

The in-repo copy is the SINGLE SOURCE OF TRUTH — it is version-controlled,
machine-checked against the codebase by tests/test_skills.py, and
auto-discovered in place by Claude Code sessions opened in this repo
(no install step needed there).

This script exists for every other surface:

  # export portable .skill bundles (zip of the skill folder) to dist/skills/
  python scripts/package_skills.py

  # copy the skills into another agent's skills directory
  python scripts/package_skills.py --install --target ~/.claude/skills

Installed/exported copies are SNAPSHOTS of the current checkout. They do
not update themselves; re-run after pulling. Prefer working inside the
repo where the skills can never drift from the code they describe.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
DEFAULT_DIST = REPO_ROOT / "dist" / "skills"


def discover_skills() -> list[Path]:
    """Every skill directory shipped with the project."""
    if not SKILLS_DIR.is_dir():
        raise SystemExit(f"skills directory missing: {SKILLS_DIR}")
    dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
    if not dirs:
        raise SystemExit(f"no skills found under {SKILLS_DIR}")
    return dirs


def validate_skill(skill_dir: Path) -> str:
    """Cheap structural check (the deep checks live in tests/test_skills.py).

    Returns the skill name from frontmatter; raises on malformed skills so
    a broken bundle can never be exported.
    """
    manifest = skill_dir / "SKILL.md"
    if not manifest.is_file():
        raise SystemExit(f"{skill_dir.name}: missing SKILL.md")
    text = manifest.read_text(encoding="utf-8")
    m = re.match(r"\A---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not m:
        raise SystemExit(f"{skill_dir.name}: SKILL.md lacks frontmatter")
    name = re.search(r"^name:\s*(\S+)\s*$", m.group(1), flags=re.MULTILINE)
    desc = re.search(r"^description:\s*(.+)$", m.group(1), flags=re.MULTILINE)
    if not name or not desc:
        raise SystemExit(f"{skill_dir.name}: frontmatter needs name + description")
    if name.group(1) != skill_dir.name:
        raise SystemExit(
            f"{skill_dir.name}: frontmatter name {name.group(1)!r} != directory name"
        )
    return name.group(1)


def package(dist_dir: Path) -> list[Path]:
    """Zip each skill folder into ``<dist>/<name>.skill`` (portable bundle:
    a zip archive whose root entry is the skill directory)."""
    dist_dir.mkdir(parents=True, exist_ok=True)
    bundles: list[Path] = []
    for skill_dir in discover_skills():
        name = validate_skill(skill_dir)
        bundle = dist_dir / f"{name}.skill"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(skill_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(skill_dir.parent)))
        bundles.append(bundle)
        try:
            shown = bundle.relative_to(REPO_ROOT)
        except ValueError:  # --dist outside the repo
            shown = bundle
        print(f"packaged {shown}")
    return bundles


def install(target: Path) -> list[Path]:
    """Copy each skill directory into ``target`` (an agent skills folder)."""
    target = target.expanduser()
    target.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    for skill_dir in discover_skills():
        name = validate_skill(skill_dir)
        dest = target / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(skill_dir, dest)
        installed.append(dest)
        print(f"installed {name} -> {dest}")
    print(
        "\nNOTE: installed copies are snapshots of this checkout; the "
        "version-controlled .claude/skills/ is the source of truth. "
        "Re-run after pulling changes."
    )
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST,
                        help="output directory for .skill bundles")
    parser.add_argument("--install", action="store_true",
                        help="copy skills into --target instead of zipping")
    parser.add_argument("--target", type=Path, default=Path("~/.claude/skills"),
                        help="skills directory to install into (with --install)")
    args = parser.parse_args(argv)
    if args.install:
        install(args.target)
    else:
        package(args.dist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
