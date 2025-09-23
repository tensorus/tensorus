#!/usr/bin/env python3
"""
Notebook validation script for Tensorus tutorials.
Validates JSON structure and nbformat compliance.
"""

import json
import sys
from pathlib import Path

def validate_notebook(notebook_path):
    """Validate a Jupyter notebook file."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        
        # Check required fields
        required_fields = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
        for field in required_fields:
            if field not in notebook_data:
                return False, f"Missing required field: {field}"
        
        # Check nbformat version
        if notebook_data.get('nbformat') != 4:
            return False, f"Expected nbformat 4, got {notebook_data.get('nbformat')}"
        
        # Validate cells structure
        cells = notebook_data.get('cells', [])
        if not isinstance(cells, list):
            return False, "Cells must be a list"
        
        for i, cell in enumerate(cells):
            if not isinstance(cell, dict):
                return False, f"Cell {i} must be a dictionary"
            
            required_cell_fields = ['cell_type', 'metadata', 'source']
            for field in required_cell_fields:
                if field not in cell:
                    return False, f"Cell {i} missing required field: {field}"
        
        return True, "Valid notebook"
        
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"

def main():
    """Main validation function."""
    tutorial_dir = Path("tutorials")
    notebooks = list(tutorial_dir.glob("*.ipynb"))
    
    print("🔍 Validating Tensorus Tutorial Notebooks")
    print("=" * 50)
    
    all_valid = True
    
    for notebook in notebooks:
        print(f"\n📓 Checking: {notebook.name}")
        is_valid, message = validate_notebook(notebook)
        
        if is_valid:
            print(f"   ✅ {message}")
        else:
            print(f"   ❌ {message}")
            all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("🎉 All notebooks are valid!")
        return 0
    else:
        print("⚠️  Some notebooks have validation errors.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
