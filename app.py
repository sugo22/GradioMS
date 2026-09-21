## This is a Gradio application for mass spectrometry data analysis and classification modeling built completely in Python with the 
## machine learning algorithms implemented in Python as well

import gradio as gr
from backend import (
    load_file,
    split_metadata,
    baseline_correction,
    two_step,
    run_pymodel
)

with gr.Blocks() as demo:

    gr.Markdown("# Mass Spectra PLS Dashboard")

    df_state = gr.State(None)
    X_state = gr.State(None)
    y_state = gr.State(None)
    X_corr_state = gr.State(None)
    X_proc_state = gr.State(None)

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
    with gr.Tab("Preprocessing"):
        m1 = gr.Dropdown(
            [
                "None",
                "Derivatives",
                "Savitzky-Golay"
            ],
            value="None",
            label="Step 1"
        )
        
        m2 = gr.Dropdown(
            [
                "None",
                "Derivatives",
                "Savitzky-Golay"
            ],
            value="None",
            label="Step 2"
        )
        
        order = gr.Slider(
            0,
            10,
            value=1,
            step=1,
            label="Derivative Order"
        )
        
        p = gr.Slider(
            1,
            5,
            value=2,
            step=1,
            label="Polynomial Order"
        )
        
        w = gr.Slider(
            3,
            21,
            value=5,
            step=2,
            label="Window Size"
        )

        m = gr.Slider(
            0,
            2,
            value=0,
            step=1,
            label="Derivative (SG)"
        )
        
        process_btn = gr.Button("Process")

        proc_plot = gr.Plot()

        process_btn.click(
            two_step,
            inputs=[
                X_corr_state,
                m1,
                m2,
                order,
                p,
                w,
                m
            ],
            outputs=[
                proc_plot,
                X_proc_state
            ]
        )
    with gr.Tab("Classification Modeling"):

        model_type = gr.Dropdown(
            choices=[
                "SVM",
                "PLS"
            ],
            value="SVM",
            label="Select Model"
        )

        train_percent = gr.Slider(
            minimum=0.5,
            maximum=0.9,
            value=0.7,
            step=0.05,
            label="Training Percentage"
        )

        preprocess = gr.Dropdown(
            choices=[
                "none",
                "scale"
            ],
            value="scale",
            label="Preprocessing"
        )

        cv_method = gr.Dropdown(
            choices=[
                "StratifiedKFold"
            ],
            value="StratifiedKFold",
            label="Cross Validation"
        )

        resampling = gr.Slider(
            minimum=3,
            maximum=10,
            value=5,
            step=1,
            label="CV Folds"
        )


        # SVM-specific options

        svm_method = gr.Dropdown(
            choices=[
                "svmLinear",
                "svmRadial",
                "svmPoly"
            ],
            value="svmRadial",
            label="SVM Kernel"
        )


        run_btn = gr.Button(
            "Run Model"
        )


        accuracy_output = gr.Number(
            label="Accuracy"
        )

        kappa_output = gr.Number(
            label="Cohen Kappa"
        )

        parameter_output = gr.JSON(
            label="Best Parameters"
        )

        confusion_output = gr.DataFrame(
            label="Confusion Matrix"
        )

        accuracy_plot = gr.Plot(
            label="Model Performance"
        )

        importance_plot = gr.Plot(
            label="Variable Importance"
        )

        run_btn.click(
            run_pymodel,
            inputs=[
                X_proc_state,
                y_state,
                model_type,
                train_percent,
                preprocess,
                cv_method,
                resampling,
                svm_method
            ],
            outputs=[
                accuracy_output,
                kappa_output,
                parameter_output,
                confusion_output,
                accuracy_plot,
                importance_plot
            ]
        )

demo.launch(inbrowser=True)