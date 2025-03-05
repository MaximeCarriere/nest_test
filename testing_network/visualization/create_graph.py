import time
from pathlib import Path
import matplotlib.pyplot as plt
import random
import pandas as pd
from collections import Counter
import numpy as np
import pandas as pd
from ast import literal_eval
from IPython.display import display
import seaborn as sns
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pickle
import glob
import warnings
#from functions.function_annexe import *
import matplotlib.patches as patches
plt.rcParams["figure.figsize"] = (20,15)
plt.rcParams["font.size"] = "15"

from config import TESTING_OUTPUT_GRAPH, NETWORKS_LIST_GRAPH, GRAPH_MODE, TESTING_OUTPUT_DIR


# Your original function to get unique values from 'nstr' column
def unique_val(df_spe_stim):
    unique_elements = set()
    for row in df_spe_stim['nstr']:
        unique_elements.update(row)  # Collect all unique elements from the list in each row
    return [k for k in unique_elements]  # return the list of unique elements

def gather_data_ca_size(pres):
    for presentation in pres:
        dfS_audi = read_tca_files(TESTING_OUTPUT_DIR, "Audi", presentation)
        dfS_arti = read_tca_files(TESTING_OUTPUT_DIR, "Arti", presentation)
        
        dfS = pd.concat([dfS_audi, dfS_arti])

    re = []
    for pres in dfS.Presentation.unique():
        dfS2 = dfS[dfS.Presentation==pres]
        data = get_ca_size(dfS2.dropna())
        data["Presentation"]=pres
        re.append(data)

    re = pd.concat(re)
        
    return re

def get_ca_size(df3):
    df3 = df3[(df3.time < 25) & (df3.time > 4)]
    # List to store the results
    data = []
    
    # Iterate through unique pattern numbers (patt_no) and areas (AreaAbs)
    for patt_no in df3.patt_no.unique():
        for area in df3.AreaAbs.unique():
            neuron_counter = Counter()  # To count occurrences of neurons across conditions

            # Calculate the number of unique condition/stimulus combinations for this patt_no and area
            unique_combinations = df3[(df3.patt_no == patt_no) & (df3.AreaAbs == area)]
            counter = unique_combinations[["stim", "Cond"]].drop_duplicates().shape[0]
            
            # Iterate through conditions and stimuli
            for stim in df3.stim.unique():  # First loop over stimulus
                for cond in df3.Cond.unique():  # Then loop over conditions
                    # Filter DataFrame based on the current pattern number, area, condition, and stimulus
                    df_spe_stim = df3[(df3.patt_no == patt_no) &
                                      (df3.AreaAbs == area) &
                                      (df3.stim == stim) &
                                      (df3.Cond == cond)]
                    
                    # If there are rows matching the filter, extract unique elements from 'nstr'
                    if len(df_spe_stim) > 0:
                        neurons = unique_val(df_spe_stim)
                        neuron_counter.update(neurons)  # Count the occurrences of neurons
            
            # Use the dynamic threshold based on the counter (unique combinations of stim and cond)
            dynamic_threshold = counter

            for thresh in range(0, df3.stim.nunique()):
                # Filter neurons that appear in more than the dynamic threshold of condition/stimulus sets
                neurons_in_all = [neuron for neuron, count in neuron_counter.items() if count > thresh]
                neuron_count = len(neurons_in_all)
                
                # Append the result to the data list
                data.append([patt_no, area, neuron_count, neurons_in_all, thresh])
        
    # Convert the results into a DataFrame with an added 'thresh' column
    re = pd.DataFrame(data, columns=["patt_no", "area", "size", "neuron", "thresh"])
    
    # Display the DataFrame
    return re







def read_tca_files(directory, cond, pres):
    directory_path = directory+cond.lower()+"_"+str(pres)+"_presentations.csv"
    try:
        df = pd.read_csv(directory_path)
    
        df = df[(df.Presentation==pres)&(df.Cond==cond)]
        df["nstr"] = df["nstr"].apply(lambda x: literal_eval(x) if "[" in x else x)

        # Define the columns you want to preserve
        key_columns = ["AreaAbs", "patt_no", "time", "stim", "Cond", "Presentation"]

        # Create all possible combinations of key columns
        idx = pd.MultiIndex.from_product(
            [df[col].unique() for col in key_columns], names=key_columns
        )

        # Convert df to have the same MultiIndex
        df = df.set_index(key_columns)

        # Merge with a DataFrame of all possible combinations, ensuring only 'sum' is filled with 0
        dfS = df.reindex(idx).reset_index()
        dfS["sum"] = dfS["sum"].fillna(0)  # Fill only the 'sum' column with 0

        # Optional: Forward fill if needed
        dfS.loc[:, dfS.columns != 'nstr'] = dfS.loc[:, dfS.columns != 'nstr'].ffill()


        return dfS
    
    except:

        print(directory_path +" not found!")

def plot_tca(dfS, cond, pres, action_object=True, save=True):

    list_area = ['V1',
     'TO',
     'AT',
     'PF_L',
     'PM_L',
     'M1_L',
     'A1',
     'AB',
     'PB',
     'PF_i',
     'PM_i',
     'M1_i']

    # Define the grid dimensions
    nrows, ncols = 2, 6
    fontsize_nb = 20

    fig_aspect = float(nrows) / float(ncols)

    # Create figure with specified aspect ratio
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    
    if action_object==True:
    # Process your data
        dfS.loc[dfS.patt_no < dfS.patt_no.nunique() / 2, "Spe"] = "Obj"
        dfS.loc[dfS.patt_no > (dfS.patt_no.nunique() / 2) - 1, "Spe"] = "Act"
        
    else:
        dfS["Spe"]="Others"
        
    dfS1 = dfS[dfS.time < 30]

    # Initialize legend handles and labels
    handles, labels = None, None

    # Plot each subplot
    for i in range(0, 12):
        ax = plt.subplot(2, 6, (i + 1))
        if len(dfS1[dfS1.AreaAbs == i]) > 0:
            lineplot = sns.lineplot(
                x="time", y="sum", hue="Spe", data=dfS1[dfS1.AreaAbs == i], linewidth=5
            )
            if i == 0:  # Capture handles and labels from the first plot
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()  # remove legend for all other plots
            else:
                ax.get_legend().remove()  # remove legend for all other plots

            plt.ylim([0, 30])
            plt.xlim([0, 20])

        plt.title(list_area[i], fontsize=25)
        plt.ylabel("")
        plt.xlabel("")
        if i == 0 or i == 6:
            plt.yticks(fontsize=25)

        else:
            plt.yticks(fontsize=0)

        if i > 5:
            plt.xticks(fontsize=20)
        else:
            plt.xticks(fontsize=0)

        # Highlight a specific region if applicable
        if cond=="Audi":
            if i == 6:
                plt.axvspan(4, 6, alpha=0.2, color='red', label='Stim Auditory')
        elif cond=="Arti":
            if i == 11:
                plt.axvspan(4, 6, alpha=0.2, color='red', label='Stim Articulatory')

    # Adjust layout and add axis labels
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85, wspace=0.1, hspace=0.1)
    fig.text(0.01, 0.5, "Number of Spike", va='center', rotation=90, fontsize=fontsize_nb + 5)
    fig.text(0.5, 0.01, "Time-step", va='center', fontsize=fontsize_nb + 5)
    plt.suptitle(cond+" Stim || "+ str(pres)+" Presentations",fontsize=25)

    # Save figure
    if save==True:
        plt.savefig(TESTING_OUTPUT_GRAPH+cond+'_TCA_'+str(pres)+'_presentations.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def plot_ca_size(re, pres=1000, save=True):
    """
    Plot the CA size over threshold for different brain areas.

    Parameters:
    - re (DataFrame): Data containing area, threshold, size, and Presentation.
    - pres (int): Number of presentations to filter data.
    - save (bool): Whether to save the plot as a file.
    """
    
    list_area = ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L',
                 'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']

    # Define the grid dimensions
    nrows, ncols = 2, 6
    fontsize_nb = 20
    
    re["Spe"]=np.nan
    re.loc[re.patt_no<re.patt_no.nunique()/2+1, "Spe"]="Obj"
    re.loc[re.patt_no>(re.patt_no.nunique()/2), "Spe"]="Act"

    thresh_plot = round(re.thresh.max()/2+1)
    fig_aspect = float(nrows) / float(ncols)
    
    # Create figure with specified aspect ratio
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))

    # Loop through the areas and create subplots
    for i in range(12):
        ax = plt.subplot(2, 6, i + 1)
        subset = re[(re.area == i) & (re.Presentation == pres)&(re.thresh==thresh_plot)]

        if len(subset) > 0:
            sns.barplot(x="Spe", y="size", hue="Spe", data=subset, ax=ax)
            plt.ylim([0, 50])
            #ax.legend_.remove()

        plt.title(list_area[i], fontsize=25)
        plt.ylabel("")
        plt.xlabel("")

        # Format y-axis labels
        if i == 0 or i == 6:
            plt.yticks(fontsize=25)
        else:
            plt.yticks([])

        # Format x-axis labels
        if i > 5:
            plt.xticks(fontsize=20)
        else:
            plt.xticks([])

    # Adjust layout and add axis labels
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.9, top=0.85, wspace=0.15, hspace=0.2)
    fig.text(0.01, 0.5, "Number of Neurons", va='center', rotation=90, fontsize=fontsize_nb + 5)
    fig.text(0.5, 0.01, "Threshold", va='center', fontsize=fontsize_nb + 5)
    plt.suptitle("CA Size || "+str(pres)+"  Presentations ||  Thresh:  "+str(thresh_plot), fontsize=25)

    # Save figure if requested
    if save:
        plt.savefig(TESTING_OUTPUT_GRAPH+"CA_Size_"+str(pres)+"_presentations.png", dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

    
    

def plot_ca_size_thresh(re, pres=1000, save=True):
    """
    Plot the CA size over threshold for different brain areas.

    Parameters:
    - re (DataFrame): Data containing area, threshold, size, and Presentation.
    - pres (int): Number of presentations to filter data.
    - save (bool): Whether to save the plot as a file.
    """
    
    list_area = ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L',
                 'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']

    # Define the grid dimensions
    nrows, ncols = 2, 6
    fontsize_nb = 20

    fig_aspect = float(nrows) / float(ncols)
    
    # Create figure with specified aspect ratio
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))

    # Loop through the areas and create subplots
    for i in range(12):
        ax = plt.subplot(2, 6, i + 1)
        subset = re[(re.area == i) & (re.Presentation == pres)]

        if len(subset) > 0:
            sns.barplot(x="thresh", y="size", hue="thresh", data=subset, ax=ax)
            plt.ylim([0, 50])
            ax.legend_.remove()

        plt.title(list_area[i], fontsize=25)
        plt.ylabel("")
        plt.xlabel("")

        # Format y-axis labels
        if i == 0 or i == 6:
            plt.yticks(fontsize=25)
        else:
            plt.yticks([])

        # Format x-axis labels
        if i > 5:
            plt.xticks(fontsize=20)
        else:
            plt.xticks([])

    # Adjust layout and add axis labels
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.9, top=0.85, wspace=0.15, hspace=0.2)
    fig.text(0.01, 0.5, "Number of Neurons", va='center', rotation=90, fontsize=fontsize_nb + 5)
    fig.text(0.5, 0.01, "Threshold", va='center', fontsize=fontsize_nb + 5)
    plt.suptitle(f"CA Size over Threshold || {pres} Presentations", fontsize=25)

    # Save figure if requested
    if save:
        plt.savefig(TESTING_OUTPUT_GRAPH+f"CA_Size_Threshold_{pres}_presentations.png", dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()



def create_interactive_plot(dfS, gi):
    """ Create an interactive 2x6 Plotly plot from the test results with individual titles for each subplot."""
    
    list_area = ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L', 'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']

    dfS = dfS[dfS.stim > 0]

    dfS["nstr"] = dfS["nstr"].apply(lambda x: literal_eval(x) if "[" in x else x)

    # Define the columns you want to preserve
    key_columns = ["AreaAbs", "patt_no", "time", "stim", "Cond"]

    # Create all possible combinations of key columns
    idx = pd.MultiIndex.from_product(
        [dfS[col].unique() for col in key_columns], names=key_columns
    )

    # Convert df to have the same MultiIndex
    dfS = dfS.set_index(key_columns)

    # Merge with a DataFrame of all possible combinations, ensuring only 'sum' is filled with 0
    dfS = dfS.reindex(idx).reset_index()
    dfS["sum"] = dfS["sum"].fillna(0)  # Fill only the 'sum' column with 0

    # Optional: Forward fill if needed
    dfS.loc[:, dfS.columns != 'nstr'] = dfS.loc[:, dfS.columns != 'nstr'].ffill()

    dfS1 = dfS[dfS.time < 30]
    if gi.empty==False:
        gi = gi.groupby(["stp", "area"]).mean().reset_index()

    # Number of rows and columns for subplots
    nrows, ncols = 2, 6

    # Create subplots with 2 rows and 6 columns, and titles based on list_area
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=list_area,
        shared_yaxes=True,
        vertical_spacing=0.1,  # Adjusted for reduced space between rows
        horizontal_spacing=0.015,  # Adjust space between columns
        # Create a secondary y-axis for GI data (right side)
        specs=[[{"secondary_y": True}]*ncols, [{"secondary_y": True}]*ncols]
    )

    # Calculate the global maximum y-value across the entire dataset for plotting
    max_y_value = dfS["sum"].max()  # This is the max value across all areas and time points

    # Define a maximum value for GI y-axis
    if gi.empty==False:
        gi_max_value = gi["GI"].max()  # Max GI value for global inhibition across all areas

    # Loop over all areas (12)
    for i, area in enumerate(list_area):
        # Filter data for the current area
        area_data = dfS[dfS['AreaAbs'] == i]  # Assuming AreaAbs is a column in dfS that denotes the area index

        # Group by 'patt_no' and 'time' to calculate the mean for each time step
        grouped_data = area_data.groupby(['patt_no', 'time']).agg(
            mean_sum=('sum', 'mean')
        ).reset_index()

        # Plot the mean for each area
        row = (i // ncols) + 1  # Calculate row for subplot
        col = (i % ncols) + 1   # Calculate column for subplot

        # Plot the mean as a line (make this the only interactive plot)
        fig.add_trace(
            go.Scatter(
                x=grouped_data["time"],
                y=grouped_data["mean_sum"],
                mode="lines",
                name=f"Mean {area}",
                line=dict(width=3, color='blue'),  # Increased thickness and blue color
                hoverinfo="x+y",  # Only show mean value when hovering
                showlegend=True if (i == 0 or i == 6) else False  # Only show legend for i = 0 and i = 6
            ),
            row=row, col=col
        )

        # Plot GI data on the second y-axis (right side)
        if gi.empty==False:
            gi_data = gi[gi['area'] == area]  # Filter GI data for the current area
            if not gi_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=gi_data["stp"],  # Assuming 'stp' is the time or step column in GI data
                        y=gi_data["GI"],  # Assuming 'GI' is the global inhibition value
                        mode="lines",
                        name=f"GI {area}",
                        line=dict(width=3, dash='dot', color="red"),
                        hoverinfo="x+y",  # Show GI values when hovering
                        showlegend=False  # Remove legend for GI plot
                    ),
                    row=row, col=col, secondary_y=True  # Use secondary y-axis
                )

        # Set the individual y-axis limits for each subplot based on the global max_y_value (spike count)
        fig.update_yaxes(
            range=[0, max_y_value * 1.1],  # Set y-limits based on the global max value
            row=row, col=col
        )
        
        if gi.empty==False:
            # Set secondary y-axis limits for GI (Global Inhibition)
            fig.update_yaxes(
                range=[0, gi_max_value * 1.1],  # Set y-limits for GI data
                secondary_y=True,
                row=row, col=col
            )

        # Set x-axis limits based on the time range
        fig.update_xaxes(
            range=[0, 30],  # Set x-limits based on the time range
            row=row, col=col
        )

        # Add y-axis title only for area 0 and 6 (Number of Spikes)

            

        # Add secondary y-axis title for GI only for i = 6 and i = 11
        if i == 5 or i == 11:
            fig.update_yaxes(
                title_text="Global Inhibition", color="red",
                secondary_y=True,
                showticklabels=True,
                row=row, col=col
            )
        else:
            fig.update_yaxes(
                secondary_y=True,
                showticklabels=False,
                row=row, col=col
            )
            
        if i == 0 or i == 6:
            fig.update_yaxes(
                title_text="Number of Spikes",color="blue",
                row=row, col=col,
                secondary_y=False,
            )


        # Add x-axis title ("Time-step") only for areas in the second row (i > 5)
        if i > 5:
            fig.update_xaxes(
                title_text="Time-step",
                row=row, col=col,
                showticklabels=True
            )
        else:
            fig.update_xaxes(
                showticklabels=False
            )

            
    # Update layout settings for square subplots and reduced space
    fig.update_layout(
        showlegend=False,  # Disable the global legend
        height=500,  # Adjusted for better readability
        width=1050,  # Ensure figure is wide enough to accommodate subplots
    )

    # Return the figure for Gradio to render
    return fig



if __name__ == "__main__":
    plot_graphs()
