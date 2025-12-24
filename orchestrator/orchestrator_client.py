
"""
Orchestrator client for the MCP code generation pipeline.

This module coordinates the entire code generation workflow:
1. Generate a high-level plan
2. Generate application code
3. Generate test code
4. Review and optionally refine the code
5. Package everything into a ZIP file
6. Return usage statistics

All MCP servers are invoked as separate processes via stdio communication.
"""

import asyncio
from pathlib import Path
from typing import Tuple, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from model_tracker import get_model_usage
from orchestrator.zip_util import create_zip_from_strings

# Project root directory (parent of orchestrator/)
BASE_DIR = Path(__file__).resolve().parents[1]


async def _call_mcp_tool(
    server_script: Path,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """
    Start an MCP server process and invoke a specific tool.
    
    This function spawns a new Python process running the MCP server script,
    establishes stdio communication, discovers available tools, and invokes
    the requested tool with the provided arguments.
    
    Args:
        server_script: Path to the Python script that implements the MCP server
        tool_name: Name of the tool to invoke (must be exposed by the server)
        arguments: Dictionary of arguments to pass to the tool
        
    Returns:
        The result returned by the tool
        
    Raises:
        ValueError: If the requested tool is not found on the server
    """
    # Configure the server to run as a Python subprocess
    server_params = StdioServerParameters(
        command="python3",
        args=[str(server_script)],
    )

    # Connect to the server via stdio (stdin/stdout)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the MCP session
            await session.initialize()
            # Discover all tools exposed by this server
            tools = await load_mcp_tools(session)

            # Find the requested tool by name
            target_tool = None
            for tool in tools:
                if tool.name == tool_name:
                    target_tool = tool
                    break

            # Validate that the tool exists
            if target_tool is None:
                available = [t.name for t in tools]
                raise ValueError(
                    f"Tool '{tool_name}' not found on server '{server_script}'. "
                    f"Available tools: {available}"
                )

            # Invoke the tool asynchronously
            result = await target_tool.ainvoke(arguments)
            return result


async def _run_pipeline_async(description: str) -> Tuple[str, Dict[str, Dict[str, int]]]:
    """
    Execute the full code generation pipeline asynchronously.
    
    Pipeline steps:
    1. Generate a high-level implementation plan
    2. Generate the application code based on the plan
    3. Generate test code for the application
    4. Review the code and tests, optionally refining if issues are found
    5. Package everything into a ZIP file
    6. Collect and return model usage statistics
    
    Args:
        description: User-provided description of the desired application
        
    Returns:
        Tuple of (zip_file_path, model_usage_dict)
    """
    # Paths to the three MCP server scripts
    refinement_server = BASE_DIR / "mcp_servers" / "refinement_server.py"
    codegen_server = BASE_DIR / "mcp_servers" / "codegen_server.py"
    testgen_server = BASE_DIR / "mcp_servers" / "testgen_server.py"

    # Step 1: Generate a high-level plan for implementation
    plan = await _call_mcp_tool(
        refinement_server,
        "generate_plan",
        {"description": description},
    )
    if not isinstance(plan, str):
        plan = str(plan)

    # Step 2: Generate the actual application code using the plan
    app_code = await _call_mcp_tool(
        codegen_server,
        "generate_app_code",
        {"description": description, "plan": plan},
    )
    if not isinstance(app_code, str):
        app_code = str(app_code)

    # Step 3: Generate test code based on the app code
    tests_code = await _call_mcp_tool(
        testgen_server,
        "generate_tests",
        {"app_code": app_code, "description": description},
    )
    if not isinstance(tests_code, str):
        tests_code = str(tests_code)

    # Step 4: Review the code and tests, then optionally refine
    feedback = await _call_mcp_tool(
        refinement_server,
        "review_code",
        {"app_code": app_code, "tests": tests_code},
    )
    if not isinstance(feedback, str):
        feedback = str(feedback)

    # If the review indicates issues (no "OK_TO_USE" marker), refine the code
    if "OK_TO_USE" not in feedback:
        refined_code = await _call_mcp_tool(
            refinement_server,
            "refine_code",
            {"app_code": app_code, "feedback": feedback},
        )
        # Only use refined code if it's valid and non-empty
        if isinstance(refined_code, str) and refined_code.strip():
            app_code = refined_code

    # Step 5: Package the generated code into a ZIP file
    zip_path = create_zip_from_strings(app_code, tests_code, description)

    # Step 6: Collect usage statistics from all model invocations
    usage = get_model_usage()

    return str(zip_path), usage


def run_pipeline(description: str) -> Tuple[str, Dict[str, Dict[str, int]]]:
    """
    Synchronous entry point for the GUI.
    
    This wrapper allows the async pipeline to be called from synchronous code
    (like the Gradio interface).
    
    Args:
        description: User-provided description of the desired application
        
    Returns:
        Tuple of (zip_file_path, model_usage_dict)
    """
    return asyncio.run(_run_pipeline_async(description))


if __name__ == "__main__":
    sample_desc = (
        "Calorie Burner is a software application that allows users to track and "
        "monitor the number of calories burned during physical activities and "
        "workouts. Users can select from a list of common activities or input "
        "custom activities to calculate the calories burned. The app provides "
        "real-time tracking of calories burned and displays an overview of the user."
    )
    zip_path, usage = run_pipeline(sample_desc)
    print("ZIP generated at:", zip_path)
    print("Model usage:", usage)
