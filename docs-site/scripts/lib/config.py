"""
Configuration for the RapidFEM docs build system.

Single package. API documentation is extracted statically (griffe,
allow_inspection=False) from the RapidFEM Python source — RapidFEM itself
does not need to be installed.
"""

from pathlib import Path

# Directory paths
SCRIPT_DIR = Path(__file__).parent.parent          # docs-site/scripts
DOCS_ROOT = SCRIPT_DIR.parent                      # docs-site
REPO_ROOT = DOCS_ROOT.parent                       # rapidfem
STATIC_DIR = DOCS_ROOT / "static"
SRC_API_DIR = DOCS_ROOT / "src" / "lib" / "api"

# The single documented package.
PACKAGE = {
    "id": "rapidfem",
    "display_name": "RapidFEM",
    # Python source location relative to the repo root.
    "source_subdir": Path("python") / "python_src",
    "root_modules": ["rapidfem"],
    "github_repo": "milanofthe/rapidfem",
}

# Only tags at or above this version are extracted. Earlier tags predate
# the current Python package layout.
MIN_SUPPORTED_VERSION = (0, 5, 0)

# Substring patterns; a module is skipped if any pattern occurs in its name.
# Excludes the CLI, native extension glue, the local UI, and example scripts.
SKIP_PATTERNS = [
    "_cli",
    "_show",
    "_native",
    "_version",
    "bridge",
    "examples",
    ".ui",
    "__pycache__",
]
