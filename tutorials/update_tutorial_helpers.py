"""Utility script to standardize tutorial notebooks.

Usage:
    python tutorials/update_tutorial_helpers.py

What it does:
- Iterates over all top-level notebooks in `tutorials/` (excluding `_archive/`).
- Inserts a shared helper import cell if `tutorial_utils.py` is not already imported.
- Skips notebooks that already contain the standardized import cell.

After running the script, review git diff to ensure cells look correct.
"""

from __future__ import annotations

import nbformat
from nbformat.notebooknode import NotebookNode
from pathlib import Path
from typing import List

TUTORIALS_DIR = Path(__file__).parent
NOTEBOOKS: List[Path] = [
    p for p in TUTORIALS_DIR.glob("*.ipynb") if p.parent.name != "_archive"
]

IMPORT_SNIPPET = (
    "from tutorial_utils import (\n"
    "    ping_server,\n"
    "    ensure_dataset,\n"
    "    ingest_tensor,\n"
    "    fetch_dataset,\n"
    "    summarize_records,\n"
    "    tensor_addition,\n"
    "    pretty_json,\n"
    ")\n"
    "API = \"http://127.0.0.1:7860\"\n"
    "SERVER = ping_server(API)\n"
    "print(f\"📡 Tensorus server available: {SERVER}\")"
)


def insert_import_cell(nb: NotebookNode) -> bool:
    """Insert the shared helper import cell if missing."""
    already_present = any(
        cell.cell_type == "code" and "from tutorial_utils import" in cell.source
        for cell in nb.cells
    )
    if already_present:
        return False

    # Insert after the first code cell (usually the dependency install cell).
    insert_idx = 0
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            insert_idx = idx + 1
            break

    import_cell = nbformat.v4.new_code_cell(IMPORT_SNIPPET)
    nb.cells.insert(insert_idx, import_cell)
    return True


def main() -> None:
    updated = []
    for notebook_path in NOTEBOOKS:
        nb = nbformat.read(notebook_path, as_version=4)
        if insert_import_cell(nb):
            nbformat.write(nb, notebook_path)
            updated.append(notebook_path.name)

    if updated:
        print("Updated notebooks:")
        for name in updated:
            print(f"  - {name}")
    else:
        print("All notebooks already contain the shared helper import.")


if __name__ == "__main__":
    main()
