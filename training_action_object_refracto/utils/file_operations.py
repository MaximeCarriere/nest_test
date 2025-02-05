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

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    neuron_pool_motor = set(range(0, 625))
    neuron_pool_visu = set(range(0, 625))
    neuron_pool_audi = set(range(0, 625))
    neuron_pool_arti = set(range(0, 625))

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
    
    print("✅ Step 4: Returning patterns")

    return motor, visu, audi, arti

