import pytest
from pathlib import Path
from nbclient import NotebookClient
import nbformat

DOC_DIR = Path(__file__).parent.parent / "doc"

@pytest.mark.parametrize("notebook_path", list(DOC_DIR.glob("*.ipynb")))
def test_notebook_execution(notebook_path):
    """Test that Jupyter notebooks execute without errors"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name='python3',
        record_timing=True,
    )
    
    client.execute()
    
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is not None, f"Cell {cell} was not executed"
            errors = [output for output in cell.get('outputs', []) if output.output_type == 'error']
            assert len(errors) == 0, f"Notebook {notebook_path} contains errors: {errors}"
