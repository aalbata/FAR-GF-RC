"""Shared bootstrap: puts the frozen sources on sys.path and resolves the project root."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src",):
    sys.path.insert(0, str(p))
def project_root(arg: str | None) -> Path:
    return (Path(arg) if arg else REPO).resolve()
