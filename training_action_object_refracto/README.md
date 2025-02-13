# 🔥 FelixNet Training Guide

This guide explains how to launch a training session, the different steps involved, and the plots generated during the process.

---

## 🚀 How to Launch Training

### **1️⃣ Activate Your Environment**
Before running the training script, ensure you activate the correct environment:

```bash
conda activate nest
2️⃣ Navigate to the Training Directory
Move to the directory where the training script is located:

bash
Copy
Edit
cd /path/to/nest_test/training_action_object_refracto
3️⃣ Run the Training
Execute the main.py script:

bash
Copy
Edit
python main.py
This will start the training process.

🛠 Steps in the Training Process
Network Initialization

FelixNet initializes 12 areas with excitatory and inhibitory neurons.
Areas are connected with specific connection rules.
Pattern Creation

Motor, visual, auditory, and articulatory patterns are generated.
Patterns are saved and used during training.
Training Process

Patterns are presented to the network in multiple repetitions.
Neurons fire according to their connections and external stimuli.
Synaptic Plasticity Updates

Weights are updated based on Hebbian learning rules.
Network stores intermediate states at defined intervals.
Final Network State

The final trained network is saved for later evaluation.
📊 Generated Plots
During training, several plots are created and saved in the plot_training directory.

1️⃣ Pattern Overlap Matrix
📌 File: plot_training/pattern_overlapp_matrix.png
Shows how much different patterns overlap.

2️⃣ Pattern Presence
📌 Files:

plot_training/motor_patterns.png
plot_training/visu_patterns.png
plot_training/audi_patterns.png
plot_training/arti_patterns.png
These plots visualize the neurons used in each pattern.

3️⃣ Activation Over Time
📌 File: plot_training/plot_activation_X.png
Shows network activation over time for each area.

4️⃣ Heatmap of Inter-Area Connections
📌 File: plot_training/heat_map_area.png
Displays the number of connections between different areas.

💾 Saving and Resuming Training
The network state is saved at different intervals (e.g., 10, 50, 100 training steps).
If interrupted, you can resume training from the last saved state.
⚠️ Common Issues
ModuleNotFoundError: No module named 'nest'

Ensure you activated the correct conda environment:
bash
Copy
Edit
conda activate nest
Plots not saving

Check if the plot_training directory exists:
bash
Copy
Edit
mkdir -p plot_training
Training is too slow

Increase the number of CPUs allocated when submitting jobs on an HPC.
🎯 Future Improvements
Adding multi-layered architecture.
Optimizing training efficiency.
Visualizing learned connections more interactively.
🚀 Happy training with FelixNet! 🚀

vbnet
Copy
Edit

Let me know if you'd like any modifications! 🚀











