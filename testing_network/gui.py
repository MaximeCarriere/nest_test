import nest
import gradio as gr
import argparse
import sys
import pickle
import pandas as pd
import plotly.express as px
from network.network import FelixNet
from visualization.create_graph import create_interactive_plot  # Import the new function

# ✅ Global FelixNet instance (updated after rebuild)
felix_net = None

if __name__ == "__main__":
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

    # ✅ Run GUI or CLI mode
    parser = argparse.ArgumentParser(description="FelixNet Testing GUI")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode without GUI")
    args = parser.parse_args()

    if args.cli:
        print("Running in command-line mode...", flush=True)
        net_file = input("Enter network file path: ")

        # ✅ Load network file safely
        with open(net_file, "rb") as f:
            network = pickle.load(f)

        audi = network["pattern_auditory"]
        arti = network["pattern_articulatory"]
        visual = network["pattern_visual"]
        motor = network["pattern_motor"]

        patt_no = int(input("Enter pattern number: "))
        num_reps = int(input("Enter number of repetitions: "))
        t_on = int(input("Enter Time ON: "))
        t_off = int(input("Enter Time OFF: "))

        # ✅ Create a new instance every time for CLI mode (safe)
        f = FelixNet()
        result = f.test_gui(audi, arti, visual, motor, patt_no, num_reps, t_on, t_off)
        print(result, flush=True)
    else:
        print("🚀 Launching FelixNet GUI...", flush=True)

        # ✅ GUI Function to Rebuild Network and Load Inputs
        def gui_rebuild_net(directory):
            """Rebuild network and extract inputs automatically."""
            print("🔄 Rebuilding Network via GUI...", flush=True)
            print(f"📂 Network File: {directory}", flush=True)

            try:
                # 🔄 Create a fresh instance
                f = FelixNet()

                # ✅ Load network data from file
                with open(directory, "rb") as f_network:
                    network = pickle.load(f_network)

                # ✅ Ensure extracted values are never None (default to empty lists)
                audi = network.get("pattern_auditory") or []
                arti = network.get("pattern_articulatory") or []
                visual = network.get("pattern_visual") or []
                motor = network.get("pattern_motor") or []

                print(f"📊 Loaded Data - Audi: {audi}, Arti: {arti}, Visual: {visual}, Motor: {motor}", flush=True)

                # ✅ Rebuild the network
                f.rebuild_net(directory)

                # ✅ Store rebuilt instance globally
                global felix_net
                felix_net = f  # Ensures correct instance is used for testing

                return "✅ Network Rebuilt Successfully!", audi, arti, visual, motor, 1
            except Exception as e:
                return f"❌ Error: {str(e)}", [], [], [], [], 1

        # ✅ Function to Run the Test and Show the Plot
        def run_test(auditory_input, articulatory_input, visual_input, motor_input, patt_no, num_reps, t_on, t_off):
            if felix_net is None:
                return "❌ Error: Network not rebuilt. Please rebuild first.", None

            result = felix_net.test_gui(auditory_input, articulatory_input, visual_input, motor_input, patt_no, num_reps, t_on, t_off)

            # Generate Plot
            fig = create_interactive_plot(pd.read_csv("./testing_gui/gui_data.csv"), "Audi", 1)
            return result, fig

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

        gui.launch(server_name="0.0.0.0", server_port=8080)
