## This is a Gradio application for mass spectrometry data analysis and classification modeling built completely in Python with the 
## machine learning algorithms implemented in Python as well

import gradio as gr
from backend import (
    load_file,
    split_metadata,
    baseline_correction
)

with gr.Blocks() as demo:

    gr.Markdown("# Mass Spectra PLS Dashboard")

    df_state = gr.State(None)
    X_state = gr.State(None)
    y_state = gr.State(None)
    X_corr_state = gr.State(None)
    with gr.Tab("Upload Data"):

        file_input = gr.File(label="Upload CSV")

        load_btn = gr.Button("Load File")

        preview_df = gr.DataFrame(label="Preview Dataset")

        metadata_selector = gr.CheckboxGroup(
            choices=[],
            label="Select Metadata Columns"
        )

        target_selector = gr.Dropdown(
            choices=[],
            label="Select Target Variable",
            interactive=True
        )
        split_btn = gr.Button(
            "Separate Metadata and Spectra"
        )

        metadata_preview = gr.DataFrame()

        spectra_plot = gr.Plot()


        load_btn.click(
            load_file,
            inputs=file_input,
            outputs=[
                df_state,
                preview_df,
                metadata_selector,
                target_selector
            ]
        )

        split_btn.click(
            split_metadata,
            inputs=[
                df_state,
                metadata_selector,
                target_selector
            ],
            outputs=[
                metadata_preview,
                X_state,
                y_state,    
                spectra_plot
            ]
        )
    with gr.Tab("Baseline Correction"):
    
            bl_method = gr.Dropdown(
                ["asls", "airpls", "modpoly"],
                value="asls",
                label="Baseline Correction Method"
            )
    
            bl_corr_btn = gr.Button(
                "Apply Baseline Correction"
            )
    
            bl_corr_plot = gr.Plot()
    
            bl_corr_btn.click(
                baseline_correction,
                inputs=[
                    X_state,
                    bl_method
                ],
                outputs=[
                    bl_corr_plot,
                    X_corr_state
                ]
            )

demo.launch(inbrowser=True)