

"""
MCP server for application code generation.

This server exposes a tool that uses an LLM to generate complete, runnable
Python applications based on user descriptions and implementation plans.
The generated code includes a Gradio GUI and follows strict requirements
for testability and completeness.
"""

from mcp.server.fastmcp import FastMCP
from typing import Optional, Tuple
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import model_tracker
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_tracker import TrackingChatGoogleGenerativeAI, _extract_text  # type: ignore

# Initialize the MCP server
mcp = FastMCP("CodeGenerator")

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
    This avoids recreating the model connection for each tool invocation.
    
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
def generate_app_code(description: str, plan: Optional[str] = None) -> str:
    """
    Generate a complete, runnable Python application from a description.
    
    Uses an LLM to generate a full Python application with:
    - Gradio GUI interface
    - Pure business logic functions (testable)
    - Standard library + gradio only
    - Complete, runnable code (no placeholders)
    
    Args:
        description: User-provided description of the desired application
        plan: Optional high-level implementation plan (if provided)
        
    Returns:
        Complete Python source code as a string, or error message if generation fails
    """
    llm, err = _get_llm()
    if err or llm is None:
        return f"ERROR: {err}"

    # Detailed system instructions for the LLM to follow
    system_instructions = """
    You are an expert Python software engineer.

    Your task is to produce a single, complete, self-contained, fully runnable Python application
    implementing the user's requirements. The final result MUST:

    1. Use ONLY the standard library plus the library 'gradio'.
    2. Include a fully functional Gradio GUI that demonstrates all features of the app.
    3. The GUI must be defined and launched using: gradio.Blocks(), gr.Interface(), or gr.Tab().
    4. The GUI must:
       - Allow the user to select from predefined activities
       - Allow custom activity name + calories-per-minute input
       - Allow the user to input duration
       - Display calories burned clearly
       - Allow logging multiple activities in a session
       - Display a summary table or running total
    5. All business logic must be placed in pure Python functions that are:
       - Deterministic
       - Easy to unit test
       - Independent of GUI state
    6. The application MUST run using: python app.py
       and MUST launch the Gradio interface on execution.
    7. Output MUST be only raw Python code — no markdown, no backticks, no explanation text.
    8. NO placeholder code, no pseudocode, no “implement here”.
    9. All numeric input components (duration, calories, etc.) MUST allow zero; use minimum=0 for all
       gr.Number or gr.Slider fields. Never set minimum to a positive value like 0.1.
    10. The overall structure, logic, and layout of the application must remain identical to the
        previously generated version unless explicitly instructed otherwise. Only fix numeric input
        validation (minimum=0).
    11. Custom activities MUST fully work. The callback for "Add to Session" MUST accept and use
        the custom activity name and custom calories-per-minute fields whenever the user selects
        the "Custom" option, and the Gradio .click() binding MUST pass these fields into the callback.
    12. The application MUST NOT use the 'height' argument (or any unsupported arguments) in
        gr.DataFrame or other Gradio components. Only use arguments supported by widely compatible
        Gradio versions (value, headers, interactive, etc.).

    FAILURE MODES TO AVOID:
    - Missing functions
    - Missing imports
    - Invalid Gradio syntax
    - Returning text instead of code
    - Using code fences
    - Writing explanations or comments that break Python syntax

    The result MUST be a COMPLETE Python file that runs as-is.
    """

    # Build the prompt with description and optional plan
    prompt_parts = [
        system_instructions.strip(),
        "\n\nApplication Description:\n",
        description.strip(),
    ]
    if plan:
        prompt_parts.append("\n\nHigh-Level Plan:\n")
        prompt_parts.append(plan.strip())

    # Invoke the LLM with the complete prompt
    prompt = "".join(prompt_parts)
    result = llm.invoke(prompt, agent_name="CodeGenerator")
    code = _get_text(result)
    return code


if __name__ == "__main__":
    mcp.run(transport="stdio")
