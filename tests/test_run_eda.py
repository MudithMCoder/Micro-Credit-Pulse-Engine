"""Test executing every code cell in notebooks/EDA.ipynb to ensure zero errors."""
import json
from pathlib import Path

nb_path = Path("notebooks/EDA.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

global_scope = {}

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_text = "".join(cell["source"])
        print(f"--- Executing Cell {idx} ---")
        try:
            exec(code_text, global_scope)
            print(f"Cell {idx} PASSED.")
        except Exception as e:
            print(f"Cell {idx} FAILED with {type(e).__name__}: {e}")
            raise e

print("\nALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY WITHOUT ERRORS.")
