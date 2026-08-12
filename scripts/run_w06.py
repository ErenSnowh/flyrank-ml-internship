import json
import io
import contextlib
import traceback
import pandas as pd
import numpy as np
import os

def execute_notebook(nb_path):
    print(f"Executing notebook top-to-bottom: {nb_path}")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    exec_globals = {}
    cell_idx = 0

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if not source.strip():
                continue

            cell_idx += 1
            output_buffer = io.StringIO()
            error_msg = None
            try:
                with contextlib.redirect_stdout(output_buffer):
                    exec(source, exec_globals)
            except Exception as e:
                error_msg = traceback.format_exc()
                print(f"  ERROR in cell {cell_idx}: {e}")

            out_text = output_buffer.getvalue()
            cell['execution_count'] = cell_idx
            cell_outputs = []
            if out_text:
                cell_outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_text.splitlines(keepends=True)
                })
            if error_msg:
                cell_outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": error_msg.splitlines(keepends=True)
                })
            cell['outputs'] = cell_outputs
            print(f"  Cell {cell_idx}: {'OK' if not error_msg else 'FAILED'}")

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Finished executing: {nb_path}\n")

if __name__ == '__main__':
    execute_notebook('work/notebooks/w06_validation_audit.ipynb')
