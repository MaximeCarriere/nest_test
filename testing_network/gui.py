import nest
import gradio as gr
import argparse
import sys
import pickle
import os
import pandas as pd
import plotly.express as px
from network.network import FelixNet
from visualization.create_graph import create_interactive_plot  # Import the new function

# ✅ Global FelixNet instance (updated after rebuild)
felix_net = None

def main():
    # ✅ Debug: Force Execution of NEST Check
    print("🔍 Checking Installed NEST Modules...", flush=True)
    installed_modules = nest.GetKernelStatus().get("loaded_modules", [])
    print(f"🔍 Installed Modules: {installed_modules}", flush=True)  # Debugging print

    if "felixmodule" not in installed_modules:
        print("⚡ Installing felixmodule...", flush=True)
        nest.Install("felixmodule")
        print("✅ Felix Module Installed!", flush=True)
    else:
        print("✅ Felix Module is already installed, skipping installation.", flush=True)

    # ✅ Initialize FelixNet once for the GUI (but not for CLI mode)
    felix_net = FelixNet()

    # ✅ Start GUI
    with gr.Blocks() as gui:
        gr.Markdown("## 🧪 FelixNet Testing GUI")

        network_file = gr.Textbox(label="Enter Network File Path")

        # 🏁 **Rebuild Network Button**
        rebuild_btn = gr.Button("Rebuild Network")
        rebuild_output = gr.Textbox(label="Rebuild Status", interactive=False)

        # ✅ **Checkboxes for Modalities (Now Editable!)**
        with gr.Row():
            auditory_check = gr.Checkbox(label="Auditory", value=False, interactive=True)
            articulatory_check = gr.Checkbox(label="Articulatory", value=False, interactive=True)
            visual_check = gr.Checkbox(label="Visual", value=False, interactive=True)
            motor_check = gr.Checkbox(label="Motor", value=False, interactive=True)

        # ✅ **Pattern Number as a Slider (Now Editable!)**
        patt_no = gr.Slider(minimum=1, maximum=12, step=1, value=1, label="Pattern Number", interactive=True)

        # ✅ Initialize hidden states for extracted inputs
        auditory_input = gr.State([])
        articulatory_input = gr.State([])
        visual_input = gr.State([])
        motor_input = gr.State([])

        # ✅ Ensure checkboxes and slider update when rebuilding, but remain interactive
        rebuild_btn.click(gui_rebuild_net,
                          inputs=[network_file],
                          outputs=[rebuild_output, auditory_input, articulatory_input, visual_input, motor_input, patt_no])

        num_reps = gr.Number(label="Number of Repetitions", value=2)
        t_on = gr.Number(label="Time ON", value=2)
        t_off = gr.Number(label="Time OFF", value=30)

        # ✅ Test button
        test_btn = gr.Button("Run test_aud")
        test_output = gr.Textbox(label="Test Status", interactive=False)

        # ✅ Plot Output (Interactive)
        plot_output = gr.Plot()

        # ✅ Run test and show plot
        test_btn.click(run_test,
                       inputs=[auditory_input, articulatory_input, visual_input, motor_input, patt_no, num_reps, t_on, t_off],
                       outputs=[test_output, plot_output])

    # ✅ Launch the Gradio app
    gui.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 8080)))

# Make sure the main function is called when the script is executed
if __name__ == "__main__":
    main()
