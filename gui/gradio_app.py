
"""
Gradio GUI application for the MCP Code Generator.

This module provides a web-based interface where users can input application
descriptions and receive generated Python applications with tests packaged in a ZIP file.
"""

import os

# Disable Gradio telemetry / analytics (and related pandas quirks)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from orchestrator.orchestrator_client import run_pipeline


def generate_app_and_tests(description: str):
    """
    Main handler function for the Gradio interface.
    
    Takes a user-provided description, runs it through the code generation pipeline,
    and returns the generated ZIP file path and model usage statistics.
    
    Args:
        description: User-provided text description of the desired application
        
    Returns:
        Tuple of (zip_file_path, model_usage_dict)
        
    Raises:
        gr.Error: If description is empty or pipeline execution fails
    """
    if not description or not description.strip():
        raise gr.Error("Please enter a description for the desired application.")
    try:
        # Call the orchestrator to run the full generation pipeline
        zip_path, usage = run_pipeline(description.strip())
    except Exception as e:
        # Surface a friendly error in the UI instead of a giant traceback
        raise gr.Error(f"Generation failed: {e}")
    return zip_path, usage


def main():
    """
    Initialize and launch the Gradio web interface.
    
    Sets up the UI components:
    - Text input for application description
    - Generate button to trigger code generation
    - File download output for the generated ZIP
    - JSON output showing model usage statistics
    """
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown("# INF119 Final Project – MCP Code Generator")
        gr.Markdown(
            "Enter the description and requirements for your desired software application. "
            "For grading, paste the Calorie Burner description here."
        )

        # Input section: text box for application description
        with gr.Row():
            description_box = gr.Textbox(
                label="Application Description",
                lines=10,
                placeholder=(
                    "Describe the application you want to generate "
                    "(e.g., Calorie Burner)..."
                ),
            )

        # Action button to trigger the generation pipeline
        generate_btn = gr.Button("Generate App & Tests")

        # Output section: ZIP file download and usage statistics
        with gr.Row():
            zip_output = gr.File(label="Download Generated ZIP")
            usage_output = gr.JSON(label="Model Usage Report (per model)")

        # Wire up the button click event to the generation function
        generate_btn.click(
            fn=generate_app_and_tests,
            inputs=description_box,
            outputs=[zip_output, usage_output],
        )

    # Launch the Gradio web server
    demo.launch()


if __name__ == "__main__":
    main()
