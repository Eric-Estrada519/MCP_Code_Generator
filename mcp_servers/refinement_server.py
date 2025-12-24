

"""
MCP server for code planning, review, and refinement.

This server provides three tools:
1. generate_plan: Creates a high-level implementation plan
2. review_code: Reviews generated code and tests for issues
3. refine_code: Improves code based on review feedback
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
mcp = FastMCP("RefinementAgent")

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
def generate_plan(description: str) -> str:
    """
    Generate a high-level implementation plan for the application.
    
    Creates a structured plan that identifies:
    - Main modules/functions/classes
    - Data flow (input, processing, output)
    - Implementation steps
    
    Args:
        description: User-provided application description
        
    Returns:
        High-level plan as plain text (no markdown)
    """
    llm, err = _get_llm()
    if err or llm is None:
        return f"ERROR: {err}"

    instructions = """
You are an expert software architect.

Given the application description, produce a concise high-level plan for how to implement it
in Python. The plan should:

- Identify the main modules/functions/classes.
- Describe how data will flow (e.g., input, processing, output).
- Be written as a short bullet list or numbered steps.
- Avoid code; focus on structure.

Return plain text (no markdown code fences).
"""
    prompt = f"""{instructions.strip()}

Application Description:
{description.strip()}
"""
    result = llm.invoke(prompt)
    plan = _get_text(result)
    return plan


@mcp.tool()
def review_code(app_code: str, tests: str) -> str:
    """
    Review generated code and tests for issues.
    
    Performs a lightweight code review checking for:
    - Missing functions or imports
    - Syntax problems
    - Test coverage adequacy
    
    If code looks acceptable, includes "OK_TO_USE" in the feedback.
    
    Args:
        app_code: The generated application code
        tests: The generated test code
        
    Returns:
        Review feedback as text (includes "OK_TO_USE" if acceptable)
    """
    llm, err = _get_llm()
    if err or llm is None:
        return f"ERROR: {err}"

    instructions = """
You are performing a lightweight code review for a classroom assignment.

Given the app code and its tests:

- Comment on obvious issues (e.g., missing functions, import mismatches, syntax problems).
- Mention whether the tests appear to meaningfully exercise the main logic.
- If the code and tests look acceptable for a basic assignment (even if not perfect),
  include the exact phrase: OK_TO_USE
"""
    prompt = f"""{instructions.strip()}

APP CODE:
{app_code}

TEST CODE:
{tests}
"""
    result = llm.invoke(prompt)
    feedback = _get_text(result)
    return feedback


@mcp.tool()
def refine_code(app_code: str, feedback: str) -> str:
    """
    Refine application code based on review feedback.
    
    Improves the code by fixing issues identified in the review while:
    - Maintaining overall structure
    - Preserving public function signatures (for test compatibility)
    - Returning only Python code (no explanations)
    
    Args:
        app_code: Current application code to refine
        feedback: Review feedback identifying issues to fix
        
    Returns:
        Refined Python source code as a string
    """
    llm, err = _get_llm()
    if err or llm is None:
        return f"ERROR: {err}"

    instructions = """
You are a Python engineer improving an existing script.

Given the current app code and textual feedback, produce an improved version
of the app code.

Requirements:
- Keep the overall structure similar, but fix obvious issues.
- Maintain the same public functions where possible so tests continue to work.
- Return ONLY the updated Python source code (no explanations or Markdown).
"""
    prompt = f"""{instructions.strip()}

CURRENT APP CODE:
{app_code}

FEEDBACK:
{feedback}
"""
    result = llm.invoke(prompt, agent_name="RefinementAgent")
    refined_code = _get_text(result)
    return refined_code


if __name__ == "__main__":
    mcp.run(transport="stdio")
