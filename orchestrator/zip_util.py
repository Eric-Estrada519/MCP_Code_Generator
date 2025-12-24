
"""
ZIP file creation utility for packaging generated code.

This module handles the creation of ZIP archives containing:
- The generated application code (app.py)
- The generated test code (test_app.py)
- A README with run instructions
"""

from pathlib import Path
from datetime import datetime
import zipfile
import shutil
import textwrap

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[1]
# Directory where generated ZIP files are stored
OUTPUT_DIR = BASE_DIR / "generated_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_zip_from_strings(app_code: str, tests_code: str, description: str) -> Path:
    """
    Create a ZIP file containing the generated application and tests.
    
    Creates a temporary directory, writes the application code, test code, and
    a README file, then packages everything into a ZIP archive. The temporary
    directory is cleaned up after ZIP creation.
    
    Args:
        app_code: The generated Python application code
        tests_code: The generated pytest test code
        description: Original application description (included in README)
        
    Returns:
        Path to the created ZIP file
    """
    # Create a unique timestamped directory for this generation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = OUTPUT_DIR / f"generated_app_{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths within the temporary directory
    app_path = temp_dir / "app.py"
    tests_path = temp_dir / "test_app.py"
    readme_path = temp_dir / "README_RUN_INSTRUCTIONS.txt"

    # Write the generated code files
    app_path.write_text(app_code)
    tests_path.write_text(tests_code)

    # Generate a README with run instructions
    instructions = textwrap.dedent(
        f"""
        Generated Application

        Description:
        {description.strip()}

        ------------------------------
        HOW TO RUN THE APP
        ------------------------------
        1. Make sure you have Python installed (3.9+ recommended).
        2. Install any required dependencies from the project root, e.g.:
           pip install -r requirements.txt
        3. From inside this directory, run:
           python app.py

        ------------------------------
        HOW TO RUN THE TESTS
        ------------------------------
        1. Install pytest if not already installed:
           pip install pytest
        2. From inside this directory, run:
           pytest test_app.py
        """
    ).strip()
    readme_path.write_text(instructions)

    # Create the ZIP archive
    zip_path = OUTPUT_DIR / f"generated_app_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add all files from the temp directory to the ZIP
        for file in temp_dir.iterdir():
            zf.write(file, arcname=file.name)

    # Clean up the temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    return zip_path
