import pickle
import time
from pathlib import Path
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sns
import re
import os
from config import *
import shutil

def ensure_directory_exists(directory, clear=False):
    """
    Ensure that the given directory exists. If it doesn't, create it.
    If `clear=True`, remove all files inside before proceeding.

    Args:
    - directory (str): The directory path to ensure exists.
    - clear (bool): If True, delete all files inside the directory before proceeding.
    """
    if os.path.exists(directory):
        if clear:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file or symbolic link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directory and its contents
                except Exception as e:
                    print(f"⚠️ Error deleting {file_path}: {e}")
        else:
            print(f"✅ Directory '{directory}' already exists. No files deleted.")
    else:
        os.makedirs(directory)
        print(f"📁 Created directory '{directory}'.")



def save_network(filename, network_data):
    ensure_directory_exists("./save_network/")
    with open(f"./save_network/{filename}", "wb") as f:
        pickle.dump(network_data, f, pickle.HIGHEST_PROTOCOL)


def show_owerlapp_pattern(motor, visu, audi, arti):

    print("#################")
    print("CHECKING OVERLAPP")
    print("#################")

    ensure_directory_exists("./plot_training")
    # Sample data (replace with your actual list of lists)
    s = {'Visual': visu, 'Motor': motor, 'Auditory': audi, 'Articulatory': arti}
    
    rows, cols = 2, 2  # Define number of rows and columns for subplots
    plot_number = 1  # Counter for subplot numbering within the grid
    
    for sys, data in s.items():
      # Calculate overlap for each pair of lists
      overlap_matrix = np.zeros((len(data), len(data)))
      for i in range(len(data)):
        for j in range(i, len(data)):  # Avoid calculating twice (i, j) and (j, i)
          overlap_matrix[i, j] = len(set(data[i]) & set(data[j]))  # Count intersection
    
      # Fill the other half of the matrix symmetrically
      overlap_matrix += overlap_matrix.T - np.diag(overlap_matrix.diagonal())
    
      # Subplot position based on plot_number
      plt.subplot(rows, cols, plot_number)
      ax = sns.heatmap(overlap_matrix, cmap='YlGnBu', annot=True)  # Annotate with values
    
      # Add labels and title (optional)
      plt.xlabel('Pattern')
      plt.ylabel('Pattern')
      plt.title(f'Overlap Heatmap - {sys}')  # Add system name to title
    
      plot_number += 1  # Increment counter for next subplot
    
    plt.tight_layout()  # Adjust spacing between subplots
    


    plt.savefig('./plot_training/pattern_overlapp_matrix.png')
    plt.close()
    

def create_act_obj_pattern(nb_pattern, size_pattern, seed=42):
    print("#################")
    print("CREATING PATTERN:")
    print("seed: "+str(seed))
    print("#################")

    # Set a fixed seed for reproducibility
    random.seed(seed)

    motor = []
    visu = []
    audi = []
    arti = []
    neuron_pool_motor = set(range(0, EXC_NEURONS*EXC_NEURONS))
    neuron_pool_visu = set(range(0, EXC_NEURONS*EXC_NEURONS))
    neuron_pool_audi = set(range(0, EXC_NEURONS*EXC_NEURONS))
    neuron_pool_arti = set(range(0, EXC_NEURONS*EXC_NEURONS))

    for i in range(nb_pattern):
        motor.append(sorted(random.sample(list(neuron_pool_motor), size_pattern)))
        visu.append(sorted(random.sample(list(neuron_pool_visu), size_pattern)))
        audi.append(sorted(random.sample(list(neuron_pool_audi), size_pattern)))
        arti.append(sorted(random.sample(list(neuron_pool_arti), size_pattern)))

        neuron_pool_motor -= set(motor[-1])
        neuron_pool_visu -= set(visu[-1])
        neuron_pool_audi -= set(audi[-1])
        neuron_pool_arti -= set(arti[-1])
        
    print("✅ Step 3: Calling `show_owerlapp_pattern()`")
    show_owerlapp_pattern(motor, visu, audi, arti)
    
    print("✅ Step 4: Calling 'plot_pattern_presence()'")
    plot_pattern_presence(motor, EXC_NEURONS, "motor_patterns")
    plot_pattern_presence(visu, EXC_NEURONS, "visu_patterns")
    plot_pattern_presence(audi, EXC_NEURONS, "audi_patterns")
    plot_pattern_presence(arti, EXC_NEURONS, "arti_patterns")
    
    print("✅ Step 4: Returning patterns")

    return motor, visu, audi, arti

def plot_pattern_presence(patterns, exc_neurons, filename):
    """
    Generates a subplot visualization of neuron presence for each pattern.

    Args:
    - patterns (list of lists): List of neuron indices for each pattern.
    - exc_neurons (int): Number of excitatory neurons per row/column.
    - filename (str): Name for the output plot image.
    """

    num_patterns = len(patterns)
    cols = min(4, num_patterns)  # Limit to 4 columns
    rows = (num_patterns // cols) + (num_patterns % cols > 0)  # Calculate required rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  # Dynamic figure size
    axes = np.array(axes).reshape(rows, cols)  # Ensure axes are in a 2D array for indexing

    for i, pattern in enumerate(patterns):
        row, col = divmod(i, cols)
        
        # Create presence matrix
        presence_matrix = np.zeros((exc_neurons, exc_neurons))
        for neuron in pattern:
            r, c = divmod(neuron, exc_neurons)
            presence_matrix[r, c] = 1  # Mark presence

        ax = axes[row, col]  # Select subplot
        sns.heatmap(presence_matrix, cmap="Blues", linewidths=0.1, linecolor="black", square=True, ax=ax)
        ax.set_title(f"Pattern {i+1}")  # Title for each pattern
        ax.set_xticks([])  # Remove axis labels for clarity
        ax.set_yticks([])

    # Hide empty subplots
    for i in range(num_patterns, rows * cols):
        fig.delaxes(axes.flatten()[i])

    # Save the figure
    output_dir = "./plot_training"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Pattern presence subplot saved: {save_path}")
