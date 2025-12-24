
"""
MCP server for test code generation.

This server generates comprehensive pytest test suites for generated applications.
The tests focus on pure business logic functions and avoid GUI dependencies.
"""

from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path
from typing import Tuple, Optional

# Ensure project root is on sys.path so we can import model_tracker
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_tracker import TrackingChatGoogleGenerativeAI, _extract_text  # type: ignore

# Initialize the MCP server
mcp = FastMCP("TestGenerator")

# Lazy-initialized LLM instance (created on first use)
_llm = None


def _get_text(result) -> str:
    """
    Extract text from an LLM result object.
    
    Wrapper around _extract_text that handles exceptions gracefully.
    
    Args:
        result: Result object from LLM invocation
        
    Returns:
        Extracted text as string
    """
    try:
        return _extract_text(result)  # type: ignore
    except Exception:
        # Fallback to string conversion if extraction fails
        return str(result)


def _get_llm() -> Tuple[Optional[TrackingChatGoogleGenerativeAI], Optional[str]]:
    """
    Lazy-initialize the LLM instance.
    
    Creates the LLM on first call and reuses it for subsequent calls.
    
    Returns:
        Tuple of (llm_instance, error_message). If successful, error_message is None.
    """
    global _llm
    if _llm is not None:
        return _llm, None
    try:
        # Use Gemini 2.5 Flash with temperature=0 for deterministic output
        _llm = TrackingChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        return _llm, None
    except Exception as e:
        return None, f"LLM initialization failed: {e}"


@mcp.tool()
def generate_tests(app_code: str, description: str) -> str:
    """
    Generate a comprehensive pytest test suite for the application.
    
    Creates test code that:
    - Tests pure business logic functions (no GUI)
    - Provides at least 10 distinct test cases
    - Covers core functionality, edge cases, and error handling
    - Returns only valid Python code (no markdown or explanations)
    
    Args:
        app_code: The application code to test
        description: Original application description (for context)
        
    Returns:
        Complete pytest test file as Python source code
    """
    llm, err = _get_llm()
    if err or llm is None:
        return f"ERROR: {err}"

    # Detailed instructions for generating clean, valid test code
    instructions = """
    You are an expert Python testing engineer.

    Your job is to generate a COMPLETE, VALID pytest test file.

    STRICT OUTPUT RULES:
    1. Output ONLY valid Python code.
    2. NO markdown of ANY kind.
    3. NO code fences (no ```python).
    4. NO explanatory text, English sentences, placeholders, or tags.
    5. NO angle brackets (< >) anywhere in the output.
    6. NO comments containing non-python tokens.
    7. NO metadata like <ctrl63>, <testcase>, </analysis>, etc.

    TESTING RULES:
    1. Import ONLY pure business logic functions from app.py.
       Example:  from app import calculate_calories, add_custom_activity
    2. Provide AT LEAST 10 real, distinct test functions.
    3. Tests MUST validate:
       - calorie calculation logic
       - built-in activities
       - custom activities
       - zero or invalid durations
       - summary/total calculations (if applicable)
    4. Tests MUST run under pytest with NO GUI logic involved.
    5. Tests MUST NOT import or run Gradio.

    STRUCTURE:
    - Begin the file with imports.
    - Then define test functions ONLY.
    - Each test must contain at least one assert statement.
    - Make all expected values realistic and consistent with the app’s logic.

    If any part of the app code appears dynamic or GUI-based, extract and test ONLY pure logic functions.
    """

    prompt = f"""{instructions.strip()}

Application Description:
{description.strip()}

Application Code:
{app_code}
"""
    result = llm.invoke(prompt, agent_name="TestGenerator")
    tests_code = _get_text(result)
    return tests_code


if __name__ == "__main__":
    mcp.run(transport="stdio")
