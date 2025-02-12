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



# Plotting helpers

def extract_events(data, time=None, sel=None):
    """Extract all events within a given time interval.

    Both time and sel may be used at the same time such that all
    events are extracted for which both conditions are true.

    Parameters
    ----------
    data : list
        Matrix such that
        data[:,0] is a vector of all gids and
        data[:,1] a vector with the corresponding time stamps.
    time : list, optional
        List with at most two entries such that
        time=[t_max] extracts all events with t< t_max
        time=[t_min, t_max] extracts all events with t_min <= t < t_max
    sel : list, optional
        List of gids such that
        sel=[gid1, ... , gidn] extracts all events from these gids.
        All others are discarded.

    Returns
    -------
    numpy.array
        List of events as (gid, t) tuples
    """
    val = []

    if time:
        t_max = time[-1]
        if len(time) > 1:
            t_min = time[0]
        else:
            t_min = 0

    for v in data:
        t = v[1]
        gid = v[0]
        if time and (t < t_min or t >= t_max):
            continue
        if not sel or gid in sel:
            val.append(v)

    return np.array(val)

def from_data(data, sel=None, **kwargs):
    """Plot raster plot from data array.

    Parameters
    ----------
    data : list
        Matrix such that
        data[:,0] is a vector of all gids and
        data[:,1] a vector with the corresponding time stamps.
    sel : list, optional
        List of gids such that
        sel=[gid1, ... , gidn] extracts all events from these gids.
        All others are discarded.
    kwargs:
        Parameters passed to _make_plot
    """
    ts = data[:, 1]
    d = extract_events(data, sel=sel)
    ts1 = d[:, 1]
    gids = d[:, 0]
    return d
    #return [ts, ts1, gids, data[:, 0]]

def from_file(fname, **kwargs):
    """Plot raster from file.

    Parameters
    ----------
    fname : str or tuple(str) or list(str)
        File name or list of file names

        If a list of files is given, the data from them is concatenated as if
        it had been stored in a single file - useful when MPI is enabled and
        data is logged separately for each MPI rank, for example.
    kwargs:
        Parameters passed to _make_plot
    """
    if isinstance(fname, str):
        fname = [fname]

    if isinstance(fname, (list, tuple)):
        try:
            global pandas
            pandas = __import__('pandas')
            return from_file_pandas(fname, **kwargs)
        except ImportError:
            from_file_numpy(fname, **kwargs)
    else:
        print('fname should be one of str/list(str)/tuple(str).')

def from_file_pandas(fname, **kwargs):
    """Use pandas."""
    data = None
    for f in fname:
        dataFrame = pandas.read_csv(
            f, sep='\s+', lineterminator='\n',
            header=None, index_col=None,
            skipinitialspace=True)
        newdata = dataFrame.values

        if data is None:
            data = newdata
        else:
            data = np.concatenate((data, newdata))
    return from_data(data, **kwargs)

#def sender2area(sender):
#    """Converts Felix sender ID to its corresponding area and ID in area"""
#    #return [neuron+(1251*area_num) for neuron in neurons]
#    for i in range(0,12):
#        ID=sender-(1251*i)
#        #print(i, sender, ID)
#        if ID <= 625:
#            area_num=i
#            return ID, area_num
#        else:
#            continue
#    #ID=sender-(1251*area_num)
#    return sender, 999


    
def sender2area(sender):
    """Converts Felix sender ID to its corresponding area and ID in area"""

    remove_pg = {0:1,
             5: 2,
            6:3,
             11:4}


    #return [neuron+(1251*area_num) for neuron in neurons]
    for i in range(0,12):
        if i in [0, 5, 6, 11]:
            add = remove_pg[i]
        else:
            add = 0

        to_remove = EXC_NEURONS*EXC_NEURONS*2+1
        ID=sender-(to_remove*i)-add
        if ID <= EXC_NEURONS*EXC_NEURONS:
            area_num=i
            return ID, area_num
        else:
            continue
    #ID=sender-(1251*area_num)
    return sender, 99999999

def convert_nstr_to_pattern(nstr):
    '''If we don't cover this in class, use this function as follows:
        First, load matplotlib with 'import matplotlib.pyplot as plt'.
        example = '137;66;34' # write your neuron string here
        pattern = convert_nstr_to_pattern(example)
        plt.imshow(pattern)'''
    # convert nstr to nID as integers
    ns=[n-1 for n in nstr]
    #ns = nstr #[int(nID)-1 for nID in (nstr_to_ns(nstr, split_char, remove_init=0))] # 'translate' felix to python
    #if len(nstr)==0:
    #    ns=nstr
    #else:
    #    ns=nstr.iloc[0]
    #    ns=[n-1 for n in ns]

    # initialize empty matrix
    # each value in the matrix = 1 neuron
    area = np.zeros((EXC_NEURONS,EXC_NEURONS))

    # write 1 for each nID at the appropriate
    # position in the matrix
    for neuron in ns:
        row = int(neuron/EXC_NEURONS)
        col = neuron%EXC_NEURONS
        area[row][col] += 1

    #area[area>1]=1
    return area


def plot_spikes(dat, time):
    nrows, ncols = 2, 6
    figsize = [8,3]
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        pattern = dat.loc[(dat["AreaAbs"]==i) & (dat["time"]==time)].matrix
        if len(pattern) != 0:
            pattern = pattern.iloc[0]
        else:
            pattern=np.zeros((25,25))
        #img = convert_nstr_to_pattern(pat_str)
        axi.imshow(pattern)
    plt.tight_layout()
    #plt.suptitle(f'{time}')
    plt.show()

def dat_from_file(string_pattern):
    """string_pattern - regex name pattern for data files to stitch together,
    i.e. 'felix-*.dat'"""
    dat = from_file([str(p) for p in Path('.').glob(string_pattern)])
    dat=pd.DataFrame(dat, columns=['sender','time','drop1','drop2'])
    dat=dat.drop(['drop1','drop2'], axis=1)
    dat=dat[pd.to_numeric(dat['sender'], errors='coerce').notnull()]
    dat=dat.reset_index(drop=True)
    dat.sender=dat.sender.astype(int)
    dat.time=dat.time.astype(float)
    dat.time=dat.time.apply(lambda x: np.round(x))
    dat.time=dat.time.astype(int)
    dat['ID']=dat.sender.apply(lambda x: sender2area(x)[0])
    dat['AreaAbs']=dat.sender.apply(lambda x: sender2area(x)[1])
    dat = dat.drop_duplicates()
    dat=dat.groupby(['AreaAbs','time'])['ID'].apply(list).reset_index(name='nstr')
    dat['matrix']=dat.nstr.apply(lambda x: convert_nstr_to_pattern(x))
    return dat


    ## Max function
def sum_arrays(row):
    return np.sum(row)


def save_plot_weight(f, pres):
    plt.rcParams["figure.figsize"] = (20,10)
    list_area_to_take = ['AT','PB','AB','A1']
    wS = []
    for area in list_area_to_take:
    
         w = np.array(f.areas[area].e_e_syn.get('weight'))
         wS.append(pd.DataFrame([len(w)*[area], len(w)*[pres], w]))

    wS = pd.concat(wS, axis=1).T
    wS.columns = ["area","pres","w"]
    wS.to_csv("./weight_data/weight_"+str(pres)+".csv")
    plt.close()

def sum_arrays(row):
    return np.sum(row)



def save_plot_activation(f, pres, dat):
    plt.rcParams["figure.figsize"] = (20,10)
    to_plot = dat.pivot(index="time", columns="AreaAbs", values="sum").fillna(0).stack().reset_index()
    custom_palette_0 = sns.color_palette("rocket", 4)  # You can choose any color palette you prefer
    custom_palette_1 = sns.color_palette("hls", 8)  # You can choose any color palette you prefer

    fig, ax = plt.subplots(2,1)
    plt.subplot(211)
    # Plot with the custom color palette
    ax = sns.lineplot(data=to_plot[to_plot.AreaAbs.isin([0, 5, 6, 11])],
                      x='time', y=0, hue="AreaAbs", linewidth=2,
                      palette=custom_palette_0)
    
    plt.xlim([to_plot["time"].min(), to_plot["time"].min()+30])
    plt.title("Stimulated Areas")
    
    plt.subplot(212)
    ax1 = sns.lineplot(data=to_plot[to_plot.AreaAbs.isin([1, 2,3,4, 7, 8, 9, 10])],
                      x='time', y=0, hue="AreaAbs", linewidth=2,
                      palette=custom_palette_1)
    
    plt.xlim([to_plot["time"].min(), to_plot["time"].min()+30])
    plt.title("Non - Stimulated Areas")

    plt.savefig('./plot_training/plot_activation_'+str(pres)+'.png')
    plt.close()



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc

def save_plot_activation_new(pres, dat, patt_no):
    plt.rcParams["figure.figsize"] = (20, 10)
    
    to_plot = dat.pivot(index="time", columns="AreaAbs", values="sum").fillna(0).stack().reset_index()
    time_tot = pd.DataFrame([k for k in range(int(to_plot.time.min()), int(to_plot.time.max()))], columns=["time"])

    #print(str(patt_no) + "   "+str(pres))
    fig, ax = plt.subplots()
    for j in range(12):
        plt.subplot(2, 6, j + 1)
        area_dat = to_plot[to_plot.AreaAbs == j]
        eph = pd.concat([area_dat.set_index("time"), time_tot.set_index("time")], axis=1)
        eph["AreaAbs"] = j
        eph[0] = eph[0].fillna(0)
        sns.lineplot(data=eph, x="time", y=0, linewidth=5)
        plt.title("Area: " + str(j) + "  Max: "+str(eph[0].max()))
        #print("Area: " + str(j) + "  Max: "+str(eph[0].max()))
        plt.xlim([to_plot.time.min(), to_plot.time.max()])
        plt.axhline(y=0, color='red', linestyle='--')
        plt.axhline(y=19, color='green', linestyle='--')
        plt.ylim([0, 80])
        plt.xlim([to_plot["time"].min(), to_plot["time"].min() + 30])
    
    plt.suptitle("Patt_no: " + str(patt_no) + '   NB Pres: ' + str(pres))
    plt.savefig('./plot_training/plot_activation_0.png')
    
    plt.clf()  # Clear the current figure
    plt.cla()  # Clear the current axes
    plt.close('all')  # Close all figures

    gc.collect()  # Run garbage collection

# Make sure to call this function appropriately within your script







def testing_auditory(audi, f):
    for patt_no in range(0, nb_pattern):
        with nest.RunManager():
            stim_specs_test = stim_specs_patt_no_testing(audi)
            self.stimulation_on(stim_specs)
            nest.Run(t_on)
        
            self.stimulation_off()
            nest.Run(t_off)

        dat=dat_from_file('felix-*.dat')
        dat['sum'] = dat['matrix'].apply(sum_arrays)
        dat.to_csv("./training_data/training_start.csv")  


def stim_specs_patt_no_testing(audi, f):
    stim_specs={'V1': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),19)),
                    'I_stim': 0},
             'M1_L': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),19)),
                    'I_stim':  0},
             'A1': {'neurons': audi[patt_no],
                    'I_stim':  stim_strength},
             'M1_i': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),19)),
                   'I_stim':  0}}

    return stim_specs
    
    
def stim_specs_noise(f, nb_noise_neuron, stim_strength_noise):
    stim_specs={'V1': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),nb_noise_neuron)),
                    'I_stim': stim_strength_noise},
             'M1_L': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),nb_noise_neuron)),
                    'I_stim':  stim_strength_noise},
             'A1': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),nb_noise_neuron)),
                    'I_stim':  stim_strength_noise},
             'M1_i': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),nb_noise_neuron)),
                   'I_stim':  stim_strength_noise}}

    return stim_specs


def stim_specs_patt_no(f, patt_no, nb_pattern, motor, visu, audi, arti, stim_strength):


    if (patt_no + 1 <= (nb_pattern/2)):
        
        stim_specs={'V1': {'neurons': visu[patt_no],
                                'I_stim': stim_strength},
                         'M1_L': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),19)),
                                'I_stim':  stim_strength},
                         'A1': {'neurons': audi[patt_no],
                                'I_stim':  stim_strength},
                         'M1_i': {'neurons': arti[patt_no],
                               'I_stim':  stim_strength}}
 

    else:

        stim_specs ={'V1': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),19)),
                            'I_stim':  stim_strength},
                     'M1_L': {'neurons': motor[patt_no],
                            'I_stim':  stim_strength},
                     'A1': {'neurons': audi[patt_no],
                            'I_stim':  stim_strength},
                     'M1_i': {'neurons': arti[patt_no],
                            'I_stim':  stim_strength}}
       

    return stim_specs
    
def stim_specs_patt_no_with_low_noise(f, patt_no, nb_pattern, motor, visu, audi, arti, stim_strength):


    noise = 3
    
    if (patt_no + 1 <= (nb_pattern/2)):
        
        
        stim_specs={'V1': {'neurons': visu[patt_no] + f.neurons2IDs(random.sample(list(range(0,625)),3)),
                                'I_stim': stim_strength},
                         'M1_L': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),19)),
                                'I_stim':  stim_strength},
                         'A1': {'neurons': audi[patt_no]+ f.neurons2IDs(random.sample(list(range(0,625)),3)),
                                'I_stim':  stim_strength},
                         'M1_i': {'neurons': arti[patt_no]+ f.neurons2IDs(random.sample(list(range(0,625)),3)),
                               'I_stim':  stim_strength}}
 

    else:

        stim_specs ={'V1': {'neurons': f.neurons2IDs(random.sample(list(range(0,625)),19)),
                            'I_stim':  stim_strength},
                     'M1_L': {'neurons': motor[patt_no]+ f.neurons2IDs(random.sample(list(range(0,625)),3)),
                            'I_stim':  stim_strength},
                     'A1': {'neurons': audi[patt_no]+ f.neurons2IDs(random.sample(list(range(0,625)),3)),
                            'I_stim':  stim_strength},
                     'M1_i': {'neurons': arti[patt_no]+ f.neurons2IDs(random.sample(list(range(0,625)),3)),
                            'I_stim':  stim_strength}}
       

    return stim_specs
    
    
def stim_specs_pre(f,  stim_strength):

    noise = 3
    visu = random.sample(list(range(0,625)),noise)
    motor = random.sample(list(range(0,625)),noise)
    audi = random.sample(list(range(0,625)),noise)
    arti = random.sample(list(range(0,625)),noise)

        
    stim_specs={'V1': {'neurons': f.neurons2IDs(visu),
                            'I_stim': stim_strength},
                     'M1_L': {'neurons': f.neurons2IDs(motor),
                            'I_stim':  stim_strength},
                     'A1': {'neurons': f.neurons2IDs(audi),
                            'I_stim':  stim_strength},
                     'M1_i': {'neurons': f.neurons2IDs(arti),
                           'I_stim':  stim_strength}}

       

    return stim_specs
    
    
    
def stim_specs_pre_new(f,  stim_strength):

    #noise_max = 3
    
    noise_visu_max = 1 if random.random() < 0.05 else 0
    noise_motor_max = 1 if random.random() < 0.05 else 0
    noise_audi_max = 1 if random.random() < 0.05 else 0
    noise_arti_max = 1 if random.random() < 0.05 else 0
    
    
    visu = random.sample(list(range(0,625)), noise_visu_max)
    motor = random.sample(list(range(0,625)),noise_motor_max)
    audi = random.sample(list(range(0,625)),noise_audi_max)
    arti = random.sample(list(range(0,625)),noise_arti_max)

        
    stim_specs={'V1': {'neurons': f.neurons2IDs(visu),
                            'I_stim': stim_strength},
                     'M1_L': {'neurons': f.neurons2IDs(motor),
                            'I_stim':  stim_strength},
                     'A1': {'neurons': f.neurons2IDs(audi),
                            'I_stim':  stim_strength},
                     'M1_i': {'neurons': f.neurons2IDs(arti),
                           'I_stim':  stim_strength}}

       

    return stim_specs



def ensure_directory_exists(directory):
    """
    Ensure that the given directory exists. If it doesn't, create it.

    Args:
    - directory: The directory path to ensure exists.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        except OSError as e:
            print(f"Error: Failed to create directory '{directory}': {e}")
    else:
        print(f"Directory '{directory}' already exists.")
        
        
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
    


    plt.savefig('./plot_training/heatmap_matrix.png')
    plt.close()




def create_act_obj_pattern(nb_pattern, size_pattern, seed=42):
    print("#################")
    print("CREATING PATTERN:")
    print("seed: "+str(seed))
    print("#################")

    print("✅ Step 1: Setting random seed")
    random.seed(seed)

    motor = []
    visu = []
    audi = []
    arti = []
    neuron_pool_motor = set(range(0, 625))
    neuron_pool_visu = set(range(0, 625))
    neuron_pool_audi = set(range(0, 625))
    neuron_pool_arti = set(range(0, 625))

    print("✅ Step 2: Generating patterns...")
    for i in range(0, nb_pattern):
        motor.append(f.neurons2IDs(sorted(random.sample(list(neuron_pool_motor), size_pattern))))
        visu.append(f.neurons2IDs(sorted(random.sample(list(neuron_pool_visu), size_pattern))))
        audi.append(f.neurons2IDs(sorted(random.sample(list(neuron_pool_audi), size_pattern))))
        arti.append(f.neurons2IDs(sorted(random.sample(list(neuron_pool_arti), size_pattern))))

        neuron_pool_motor -= set(motor[-1])
        neuron_pool_visu -= set(visu[-1])
        neuron_pool_audi -= set(audi[-1])
        neuron_pool_arti -= set(arti[-1])

    print("✅ Step 3: Calling `show_owerlapp_pattern()`")
    show_owerlapp_pattern(motor, visu, audi, arti)
    
    print("✅ Step 4: Returning patterns")
    return motor, visu, audi, arti


    
    




