#!/usr/bin/env python3
"""
Package paper/ into a minimal tarball for arXiv: LaTeX source, the compiled bibliography (main.bbl, so build
the paper first), figures and tables. arXiv builds its own PDF, so paper/main.pdf is left out.

Usage:
    python scripts/package_arxiv.py        -> dist/ln-upgrade-dynamics-arxiv.tar.gz
"""

import sys
import tarfile
from pathlib import Path

ALLOWED_EXTS = {".tex", ".bbl", ".bib", ".sty", ".cls", ".bst", ".pdf", ".png", ".jpg", ".eps"}


def main():
    repo_root = Path(__file__).resolve().parent.parent
    latex_dir = repo_root / "paper"
    if not (latex_dir / "main.bbl").exists():
        print("Error: paper/main.bbl not found; build the paper first (paper/build).", file=sys.stderr)
        sys.exit(1)

    dist_dir = repo_root / "dist"
    dist_dir.mkdir(exist_ok=True)
    tarball_path = dist_dir / f"{repo_root.name}-arxiv.tar.gz"
    print(f"Packaging {latex_dir} -> {tarball_path}")

    with tarfile.open(tarball_path, "w:gz") as tar:
        for file_path in sorted(latex_dir.rglob("*")):
            arcname = file_path.relative_to(latex_dir)
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTS and arcname != Path("main.pdf"):
                print(f"  + {arcname.as_posix()}")
                tar.add(file_path, arcname=arcname.as_posix())

    print(f"Created {tarball_path} ({tarball_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
