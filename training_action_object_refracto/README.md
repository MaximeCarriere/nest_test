# 🔥 FelixNet Training Guide

This guide explains how to launch a training session, the different steps involved, and the plots generated during the process.

---

## 🚀 How to Launch Training

### **1️⃣ Activate Your Environment**
Before running the training script, ensure you activate the correct environment:

```
conda activate nest
```

2️⃣ Navigate to the Training Directory
Move to the directory where the training script is located:

```
cd ~/nest_test/training_action_object_refracto
```

3️⃣ Run the Training
Execute the main.py script:

```
python main.py
```

## 🛠 Steps in the Training Process

### **1️⃣ Network Initialization**
- FelixNet initializes **12 areas** with excitatory and inhibitory neurons.
- Areas are connected with specific connection rules.

### **2️⃣ Pattern Creation**
- **Motor, visual, auditory, and articulatory patterns** are generated.
- Patterns are saved and used during training.

### **3️⃣ Training Process**
- Patterns are presented to the network in multiple repetitions.
- Neurons fire according to their connections and external stimuli.

### **4️⃣ Synaptic Plasticity Updates**
- Weights are updated based on Hebbian learning rules.
- Network stores intermediate states at defined intervals.

### **5️⃣ Final Network State**
- The final trained network is saved for later evaluation.

---

## 📊 Generated Plots

During training, several plots are created and saved in the `plot_training` directory.

### **1️⃣ Pattern Overlap Matrix**
- **File:** `plot_training/pattern_overlapp_matrix.png`
- **Description:** Shows how much different patterns overlap.

![pattern_overlapp_matrix](https://github.com/user-attachments/assets/249f5e6d-654e-4678-953c-f38e8dfb0bd0)

### **2️⃣ Pattern Presence**
- **Files:**
  - `plot_training/motor_patterns.png`
  - `plot_training/visu_patterns.png`
  - `plot_training/audi_patterns.png`
  - `plot_training/arti_patterns.png`
- **Description:** These plots visualize the neurons used in each pattern.

  ![arti_patterns](https://github.com/user-attachments/assets/33fd3c07-e795-442f-8c5a-71bc7f97958b)


### **3️⃣ Activation Over Time**
- **File:** `plot_training/plot_activation_X.png`
- **Description:** Shows network activation over time for each area.

![plot_activation_0](https://github.com/user-attachments/assets/7070d99b-ee27-4485-81a7-6e6fc51323c2)



### **4️⃣ Heatmap of Inter-Area Connections**
- **File:** `plot_training/heat_map_area.png`
- **Description:** Displays the number of connections between different areas.

![heat_map_area](https://github.com/user-attachments/assets/0a9c706e-aef7-439c-8b41-7386652a091f)



## 💾 Saving and Resuming Training
- The network state is saved at different intervals (e.g., 10, 50, 100 training steps).
- If interrupted, you can **resume** training from the last saved state.



