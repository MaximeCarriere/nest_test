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
import nest
import pickle
import glob
import warnings
from functions.function_annexe import *

warnings.simplefilter(action="ignore")

nest.set_verbosity("M_ERROR")

nest.Install('felixmodule')


import scipy.sparse as sp
def get_matrix_weight(df):
    # Maximum value for source and target
    max_val = max(df['source'].max(), df['target'].max())
    min_val = min(df['source'].min(), df['target'].min())

    value_to_take = max_val - min_val
    
    # Create a sparse matrix with size (max_val+1, max_val+1)
    matrix = sp.csr_matrix((df['weight'], (df['source']-min_val, df['target']-min_val)), shape=(value_to_take+1, value_to_take+1)).toarray()
    return matrix


def stim_specs_patt_no(patt_no, nb_pattern, motor, visu, audi, arti, stim_strength):


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






def stim_specs_patt_no_testing_audi_only(audi, patt_no, stim_strength):

    stim_specs={'A1': {'neurons': audi[patt_no],
                            'I_stim':   stim_strength}}

    return stim_specs
    
    
def stim_specs_patt_no_testing_arti_only(arti, patt_no, stim_strength):

    stim_specs={'M1_i': {'neurons': arti[patt_no],
                            'I_stim':   stim_strength}}

    return stim_specs


def stim_specs_patt_no_testing_visual_motor_only(visual, motor, patt_no, stim_strength):

    if (patt_no + 1) <= (len(visual) / 2):

        stim_specs = {
                'V1': {'neurons': visual[patt_no],
                       'I_stim': stim_strength}}
    else:
        stim_specs = {
                        'M1_L': {'neurons': motor[patt_no],
                            'I_stim': stim_strength}}


    return stim_specs
    
    
    
def stim_specs_pre(stim_strength):

    noise = 1
    add_noise_external = 2
    
    visu = sorted(random.sample(list(range(0,625)),noise+add_noise_external))
    motor = sorted(random.sample(list(range(0,625)),noise+add_noise_external))
    audi = sorted(random.sample(list(range(0,625)),noise+add_noise_external))
    arti = sorted(random.sample(list(range(0,625)),noise+add_noise_external))
    ab = sorted(random.sample(list(range(0,625)),noise))
    pb = sorted(random.sample(list(range(0,625)),noise))
    pf_i = sorted(random.sample(list(range(0,625)),noise))
    pm_i = sorted(random.sample(list(range(0,625)),noise))
    

        
    stim_specs={'V1': {'neurons': visu,
                            'I_stim': stim_strength},
                     'M1_L': {'neurons': motor,
                            'I_stim':  stim_strength},
                     'A1': {'neurons': audi,
                            'I_stim':  stim_strength},
                     'AB': {'neurons': ab,
                            'I_stim':  stim_strength},
                     'PB': {'neurons': pb,
                            'I_stim':  stim_strength},
                     'M1_i': {'neurons': arti,
                           'I_stim':  stim_strength},
                     'PM_i': {'neurons': arti,
                           'I_stim':  stim_strength},
                     'PF_i': {'neurons': pf_i,
                           'I_stim':  stim_strength}}

       

    return stim_specs



def stim_specs_patt_no_testing_audi_and_artionly(audi, arti, patt_no, stim_strength):

    stim_specs={'A1': {'neurons': audi[patt_no],
                            'I_stim':   stim_strength},
               'M1_i': {'neurons': arti[patt_no],
                            'I_stim':   stim_strength},
               }

    return stim_specs

def create_act_obj_pattern(nb_pattern):
    motor = []
    visu = []
    audi = []
    arti = []
    for i in range(0, nb_pattern):
        motor.append(f.neurons2IDs(sorted(random.sample(list(range(0,625)),19))))
        visu.append(f.neurons2IDs(sorted(random.sample(list(range(0,625)),19))))
        audi.append(f.neurons2IDs(sorted(random.sample(list(range(0,625)),19))))
        arti.append(f.neurons2IDs(sorted(random.sample(list(range(0,625)),19))))


    return motor, visu, audi, arti


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



#    With Spiking Neurons and Brain-Like Connectivity.
#    Front. Comput. Neurosci. 12:88.
#    DOI: https::10.3389/fncom.2018.00088
#
# Only some of the inter-area connections shown in Fig 1B are included.


        
        
        
class FelixNet:
    """
    Class representing entire network.
    """
    
    def __init__(self):
        nest.ResetKernel()
        nest.set(resolution=0.5, local_num_threads=16, rng_seed=12345)

    def connect_areas(self):
        """
        Create inter-area connections.
        """

        # As all connections are symmetric, we only specify them once as source-target pair
        # and then add the opposite connections afterwards automatically.
        connectome = [
            # Visual System
            ('V1', 'TO'),
            ('V1', 'AT'),
            ('TO', 'AT'),
            # Motor System
            ('PF_L', 'PM_L'),
            ('PF_L', 'M1_L'),
            ('PM_L', 'M1_L'),
            # Auditory System
            ('A1', 'AB'),
            ('A1', 'PB'),
            ('AB', 'PB'),
            # Articulatory System
            ('M1_i','PM_i'),
            ('M1_i','PF_i'),
            ('PM_i','PF_i'),
            # Cross system next-neighbour
            ('AT', 'PF_L'),
            ('AT', 'PB'),
            ('AT', 'PF_i'),
            ('PF_L', 'PB'),
            ('PF_L', 'PF_i'),
            ('PB', 'PF_i'),
            # Cross system jumping links
            ('TO', 'PF_L'),
            ('AT', 'PM_L'),
            ('AB', 'PF_i'),
            ('PB', 'PM_i'),

        ]
        # must create list here to avoid infinite loop
        connectome.extend(list((tgt, src) for src, tgt in connectome))

        # Now create connections
        for src, tgt in connectome:
            nest.Connect(self.areas[src].exc, self.areas[tgt].exc,
                        {'rule': 'pairwise_bernoulli',
                        'p': 0.28* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        nest.spatial.distance.y,
                        std_x=6.5,
                         std_y=6.5,
                         mean_x = 1,
                         mean_y= 1,
                         rho =0.70710678
                             ),
                          'mask': {'grid': {'shape': [19, 19]}, 'anchor': [9, 9]}},
                         {'synapse_model': 'abs_synapse', 'receptor_type': 1,
                         'weight': nest.random.uniform(between_min, between_max), 'delay': 0.5}) # orig 'weight': nest.random.uniform(0, 0.1)


    def reconnect_areas(self, directory_network):
        """
        Create inter-area connections.
        """

        with open(directory_network, "rb") as f:
            network = pickle.load(f)

        # As all connections are symmetric, we only specify them once as source-target pair
        # and then add the opposite connections afterwards automatically.
        connectome = [('V1', 'TO'),
                      ('V1', 'AT'),
                      ('TO', 'AT'),
                      ('TO', 'PF_L'),
                      ('AT', 'PF_L'),
                      ('AT', 'PB'),
                      ('AT', 'PF_i'),
                      ('AT', 'PM_L'),
                      ('PF_L', 'PM_L'),
                      ('PF_L', 'PB'),
                      ('PF_L', 'PF_i'),
                      ('PF_L', 'M1_L'),
                      ('PM_L', 'M1_L'),
                      ('A1', 'AB'),
                      ('A1', 'PB'),
                      ('AB', 'PB'),
                      ('AB', 'PF_i'),
                      ('PB', 'PF_i'),
                      ('PB', 'PM_i'),
                      ('PF_i', 'PM_i'),
                      ('PF_i', 'M1_i'),
                      ('PM_i', 'M1_i'),
                     ]
        # must create list here to avoid infinite loop
        # must create list here to avoid infinite loop
        connectome.extend(list((tgt, src) for src, tgt in connectome))


         ## Do you need static or learnable connections?
        connection_type = "static_synapse"
        #connection_type = "abs_synapse"
         

        excitatory_neurons = network["excitatory_neurons"]
        
        # Now create connections
        for src, tgt in connectome:

            area1 = src
            area2 = tgt
            excitatory_neurons_area1 = excitatory_neurons[excitatory_neurons.area==area1]
            global_id_area1_exc = excitatory_neurons_area1.global_id.unique().tolist()

            excitatory_neurons_area2 = excitatory_neurons[excitatory_neurons.area==area2]
            global_id_area2_exc = excitatory_neurons_area2.global_id.unique().tolist()

            weight_area_exc = network["weight"][(network["weight"].source.isin(global_id_area1_exc))&(network["weight"].target.isin(global_id_area2_exc))]
#            n_area1 = excitatory_neurons[excitatory_neurons["area"]==area1]
#            n_area2 = excitatory_neurons[excitatory_neurons["area"]==area2]
#            max_value = max(n_area1["global_id"].nunique(), n_area2["global_id"].nunique())
#            
#            
#            min_area1 = n_area1["global_id"].min()
#            min_area2 = n_area2["global_id"].min()
#            
#            weight = network["weight"]
#            weight_to_keep_a1_source_a2_target = weight[(weight.source.isin(n_area1.global_id.unique().tolist()))&
#                    (weight.target.isin(n_area2.global_id.unique().tolist()))]
#            
#            weight_to_keep_a2_source_a1_target = weight[(weight.source.isin(n_area2.global_id.unique().tolist()))&
#                    (weight.target.isin(n_area1.global_id.unique().tolist()))]
#            
#            
#            weight_to_keep_a1_source_a2_target["source"] = weight_to_keep_a1_source_a2_target["source"] - (min_area1-1)
#            weight_to_keep_a2_source_a1_target["source"] = weight_to_keep_a2_source_a1_target["source"] - (min_area2-1)
#            
#            weight_to_keep_a1_source_a2_target["target"] = weight_to_keep_a1_source_a2_target["target"] - (min_area2-1)
#            weight_to_keep_a2_source_a1_target["target"] = weight_to_keep_a2_source_a1_target["target"] - (min_area1-1)
#            
#            weight_to_keep_tot = pd.concat([weight_to_keep_a2_source_a1_target, weight_to_keep_a1_source_a2_target],axis = 0)
#            weight_to_keep_tot = get_matrix_weight(weight_to_keep_tot)
#            weight_to_keep_tot = weight_to_keep_tot

            #weight_to_keep_tot[weight_to_keep_tot == 0] = np.nan

           
#            nest.Connect(self.areas[src].exc, self.areas[tgt].exc,
#                        "all_to_all",
#                         {'synapse_model': connection_type,
#                         'receptor_type': 1,
#                         'weight': weight_to_keep_tot,
#                         'delay': 1}) # orig 'weight': nest.random.uniform(0, 0.1)
#                         
            nest.Connect(weight_area_exc["source"].values, weight_area_exc["target"].values,
                             "one_to_one",
                             {"synapse_model":connection_type,
                              "weight": weight_area_exc["weight"].values,
                             "delay":weight_area_exc["delay"].values,
                              'receptor_type': 1
                             })#,
    


    
    def connect_recorders(self):
        """
        Connect spike recorder.
        """

        self.spike_rec = nest.Create('felix_spike_recorder',
                                     {'record_to': 'ascii', 'label': 'felix'})
        self.vm = nest.Create('multimeter', params={'record_from': ["V_m", "I_tot",  "I_exc","I_inh", "I_noise"],
                                                    'record_to': 'ascii', 'label': 'V_m'})

        #self.mi = nest.Create('multimeter', params={'record_from': ["V_m", "I_tot"]})
        #self.mg = nest.Create('multimeter', params={'record_from': ["V_m", "I_tot"]})
        
        for area in self.areas.values():
            nest.Connect(area.exc, self.spike_rec)
            nest.Connect(self.vm, area.exc)
            #nest.Connect(self.mi, self.inh)
            #nest.Connect(self.mg, self.glob)
            
            
    def build_net(self):
        """
        Build network
        1. All areas with internal connections
        2. Connections between areas
        3. Connect recorders
        """
        
        self.areas = {area: Area() for area in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L',
                                                'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']}
        self.connect_areas()
        self.connect_recorders()

    def stimulation_on(self, stim_specs):
        for area, specs in stim_specs.items():
            self.areas[area].stimulation_on(**specs)

    def stimulation_off(self):
        for area in self.areas.values():
            area.stimulation_off()

    def neurons2IDs(self, neurons):
        """
        Translates neuron numbers from 1 to 625 to their corresponding ID within an area of the model.

        neurons -- list of neuron numbers
        area_num -- ID of area in model structure
        """
        #return [neuron+(1251*area_num) for neuron in neurons]
        return sorted(neurons)
        
    def train(self, num_reps=10, t_on=16, t_off=84, stim_specs=None):
        """
        Run Felix training protocol.

        num_reps -- number of training cycles
        t_on -- number of timesteps during which stimulus is presented per cycle
        t_off -- number of time steps during which stimulus is off per cycle
        stim_specs -- dictionary specifying who and how to stimulate
                      - top level maps area names to dictionaries with details
                      - for each area {'neurons': [...], 'I_stim': n.nn}
                        where [...] is list of neuron indices in area and n.nn is
                        current amplitude for stimulation
        """
        
        with nest.RunManager():
            for _ in range(num_reps):
                self.stimulation_on(stim_specs)
                nest.Run(t_on)
                
                self.stimulation_off()
                nest.Run(t_off)

    
    def test(self, num_reps=1, t_on=2, t_off=50, stim_specs=None):
        """
        Run Felix testing protocol.

        TODO: turn off synaptic plasticity

        num_reps -- number of training cycles
        t_on -- number of timesteps during which stimulus is presented per cycle
        t_off -- number of time steps during which stimulus is off per cycle
        stim_specs -- dictionary specifying who and how to stimulate
                      - top level maps area names to dictionaries with details
                      - for each area {'neurons': [...], 'I_stim': n.nn}
                        where [...] is list of neuron indices in area and n.nn is
                        current amplitude for stimulation
        """
        
        with nest.RunManager():
            for _ in range(num_reps):
                self.stimulation_on(stim_specs)
                nest.Run(t_on)
                
                self.stimulation_off()
                nest.Run(t_off)


    def store(self, dump_filename, motor, visu, audi, arti):
        """
        Store neuron membrane potential and synaptic weights to given file.
        """

        assert nest.NumProcesses() == 1, "Cannot dump MPI parallel"

        ## Inhibitory Parameters
        inh_neurons_param_list = ["k_1", "tau_m"]
        list_param_value_inh = []
        for param in inh_neurons_param_list:
            value = list(set(self.areas['A1'].inh.get([param], output="pandas")[param].values.tolist()))
            list_param_value_inh.append([param, value])
        list_param_value_inh = pd.DataFrame(list_param_value_inh, columns=["param", "value"])

        ## Excitatory Parameters
        exc_neurons_param_list = ["om", "alpha", "alpha_e", "tau_adapt", "k_2", "Jexcitatory"]
        list_param_value = []
        for param in exc_neurons_param_list:
            value = list(set(self.areas['A1'].exc.get([param], output="pandas")[param].values.tolist()))
            list_param_value.append([param, value])
        list_param_value = pd.DataFrame(list_param_value, columns=["param", "value"])

        ## Global Inhibition Parameters
        glob_neurons_param_list = ["k_1", "tau_m"]
        list_param_value_glob = []
        for param in glob_neurons_param_list:
            value = list(set(self.areas['A1'].glob.get([param], output="pandas")[param].values.tolist()))
            list_param_value_glob.append([param, value])
        list_param_value_glob = pd.DataFrame(list_param_value_glob, columns=["param", "value"])

        ## Store excitatory, inhibitory and global inhibition values

        excitatory_neurons = []
        inhibitory_neurons = []
        global_inhibition = []
        
        for area in self.areas.keys():
            eph = self.areas[area].exc.get(output="pandas")
            eph["area"] = area
            excitatory_neurons.append(eph)
        
            eph_i = self.areas[area].inh.get(output="pandas")
            eph_i["area"] = area
            inhibitory_neurons.append(eph_i)
        
            eph_g = self.areas[area].glob.get(output="pandas")
            eph_g["area"] = area
            global_inhibition.append(eph_g)
        
        excitatory_neurons = pd.concat(excitatory_neurons)
        inhibitory_neurons = pd.concat(inhibitory_neurons)
        global_inhibition = pd.concat(global_inhibition)


        ## Store all weights
        test = nest.GetConnections().get(
                    ("delay", "receptor", "source", "synapse_model", "target", "weight"), output="pandas"
                )

        network = {}
        network["param_excitatory"] = list_param_value
        network["param_inhibitory"] = list_param_value_inh
        network["param_global"] = list_param_value_glob
        network["weight"] = test
        network["pattern_motor"] = motor
        network["pattern_visual"] = visu
        network["pattern_auditory"] = audi
        network["pattern_articulatory"] = arti
        network["excitatory_neurons"] = excitatory_neurons
        network["inhibitory_neurons"] = inhibitory_neurons
        network["global_inhibition"] = global_inhibition
                
        filename = "network_test"
        directory = "/Users/maximecarriere/src/felix-module/examples/save_network/"
        
        with open(directory+filename, "wb") as f:
                    pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)


    def rebuild_net(self, directory ):
        """
        Build network
        1. All areas with internal connections
        2. Connections between areas
        3. Connect recorders
        """
        
        #self.areas = {area: Restore_Area(directory, area) for area in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L',
        #                                        'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']}
        self.areas = {area: Restore_Area(directory, area) for area in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L',
                                                'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']}
#

        self.reconnect_areas(directory)

        self.connect_recorders()

    def train_action_object(self, motor, visu, audi, arti, num_reps=10, t_on=16, t_off=84, stim_specs=None, nb_pattern=2):

        ensure_directory_exists("./training_data")
        ensure_directory_exists("./plot_weight")
        ensure_directory_exists("./weight_data")
        ensure_directory_exists("./testing_data")
        #ensure_directory_exists("./training_nest")

        
        patt_no_count = [0]*nb_pattern
        count_firs_pres = 0
        while any(count < num_reps for count in patt_no_count):
            with nest.RunManager():
                
                ## Randomly pick a number
                patt_no = random.randint(0, nb_pattern-1)

                ## Make sure patt_no hasn't been presented too many times
                if patt_no_count[patt_no]>=num_reps:
                    pass

                else:
               
                    print("Presentation patt_no: "+str(patt_no))
                    
                    nest.SetKernelStatus({'overwrite_files': True})
                
                    ## Get the associated specificity (pattern)
                    stim_specs = stim_specs_patt_no(patt_no, nb_pattern, motor, visu, audi, arti, stim_strength)
                
                    ## Count presentation
                    patt_no_count[patt_no] += 1
                
                    print("Presentation Count: "+str(patt_no_count))
                    
                
                    self.stimulation_on(stim_specs)
                    nest.Run(t_on)
                
                    self.stimulation_off()
                    nest.Run(t_off)
                

                if patt_no_count[-1]%5==0:
                    dat=dat_from_file('felix-*.dat')
                    dat['sum'] = dat['matrix'].apply(sum_arrays)
                    dat["Pres"] = patt_no_count[-1]
                    dat["patt_no"] = patt_no
                    dat.to_csv("./training_data/training_"+str(patt_no_count[-1])+"_presentations.csv")
                    save_plot_weight(patt_no_count[-1])
                    save_plot_activation(patt_no_count[-1], dat)

                    ## AUDITORY TESTING
                    if patt_no == nb_pattern-1:
                        dataS = []
                        #time.sleep(5)
                        #target_dir = "./"
                        #remove_felix_dat_files(target_dir)
                        print("TESTING")
                        
                        for patt_no in range(0, nb_pattern):
                            nest.SetKernelStatus({'overwrite_files': True})
                            print("patt_no: "+str(patt_no))
                            #stim_specs_test = stim_specs_patt_no_testing_audi(audi, patt_no)
                            stim_specs = stim_specs_patt_no(patt_no, nb_pattern, motor, visu, audi, arti, stim_strength)
                            
                            self.stimulation_on(stim_specs)
                            nest.Run(t_on)
                            self.stimulation_off()
                            nest.Run(t_off)
            
                            dat = dat_from_file('felix-*.dat')
                            dat['sum'] = dat['matrix'].apply(sum_arrays)
                            dat["patt_no"]=patt_no
                            dat["time"] = dat["time"] - dat["time"].min()
                            dataS.append(dat)
                            
                            
                        pd.concat(dataS).to_csv("./testing_data/training_"+str(patt_no_count[-1])+"_presentations.csv")
                                              
                        
                    
                    

                #elif count_firs_pres==0:
                #    dat=dat_from_file('felix-*.dat')
                #    dat['sum'] = dat['matrix'].apply(sum_arrays)
                #    dat["Pres"] = 0
                #    dat["patt_no"] = patt_no
                #    dat.to_csv("./training_data/training_start.csv")
                #    save_plot_weight(patt_no_count[-1])
                #s    save_plot_activation(patt_no_count[-1], dat)
                    
                #    count_firs_pres = 1


        dat['sum'] = dat['matrix'].apply(sum_arrays)
        dat["Pres"] = num_reps
        dat.to_csv("./training_data/training_end.csv")
        save_plot_weight(patt_no_count[-1])
        save_plot_activation(patt_no_count[-1], dat)

        


    def test_aud(self, audi, arti,  patt_no_count, num_reps=10, t_on=16, t_off=30):
        ensure_directory_exists("./testing_data")
        ensure_directory_exists("./spikes")
        stim_strength = 500
        nest.SetKernelStatus({'overwrite_files': True})

        dataS = []
        spikes_tot  = []
        print("TESTING")
        for stim in range(0, 4):
            print("      STIM:   "+str(stim))
            for patt_no in range(0, len(audi)):
            
                run_twice = (stim == 0 and patt_no == 0)
                iterations = 2 if run_twice else 1
                
                for i in range(iterations):
                    with nest.RunManager():
                        nest.SetKernelStatus({'overwrite_files': True})
                        
                        stim_specs_test = stim_specs_patt_no_testing_audi_only(audi, patt_no, stim_strength)
                        
    #                    for i in range(0, 5):
    #                        stim_specs_pre_1 = stim_specs_pre(50)
    #                        self.stimulation_on(stim_specs_pre_1)
    #                        nest.Run(1)
    #                        self.stimulation_off()
    #                        nest.Run(2)
                            
                        #self.stimulation_off()
                        #nest.Run(2)
                        
                        print("patt_no: "+str(patt_no))
                        stim_specs_test = stim_specs_patt_no_testing_audi_only(audi, patt_no, stim_strength)
                        #stim_specs_test = stim_specs_patt_no(patt_no, nb_pattern, audi, audi, audi, audi,stim_strength)
            
                        
                        self.stimulation_off()
                        nest.Run(5)
                        
                        self.stimulation_on(stim_specs_test)
                        nest.Run(t_on)
                        
                        counter_stim_pause = 0
                        self.stimulation_off()
                        
                        gi_AB = self.areas["AB"].glob.get(output="pandas")["V_m"].values[0]
                        gi_PM_i = self.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]
                        gi_PB = self.areas["PB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        gi_PF_i = self.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        
                        print("gi_AB:", gi_AB)
                        print("gi_PM_i:", gi_PM_i)
                        #while ((gi_AB > 0.75) | (gi_PM_i >0.75) | (counter_stim_pause < 60)):
                        while ((gi_AB > 0.75) | (gi_PM_i >0.75) | (counter_stim_pause < t_off)):
                            # Print values before update


                            nest.Run(0.5)
                            gi_AB = self.areas["AB"].glob.get(output="pandas")["V_m"].values[0]
                            gi_PM_i = self.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]
                            gi_PB = self.areas["PB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                            gi_PF_i = self.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
    #                        for area_eph in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L','A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']:
    #                            spikes_eph = np.sum(f.areas[area_eph].exc.get("phi", output="pandas")["phi"].values)
    #                            spikes_tot.append([counter_stim_pause, area_eph,  spikes_eph, stim,patt_no])
    #
                            
                            counter_stim_pause += 0.5

                        # Print values after update
                        print("gi_AB:", gi_AB)
                        print("gi_PM_i:", gi_PM_i)
                        print("counter_stim_pause:", counter_stim_pause)
                        print("#######################")
            
            
                        if iterations==1:
                            dat = dat_from_file('felix-*.dat')
                            dat['sum'] = dat['matrix'].apply(sum_arrays)
                            dat["stim"] = stim
                            dat["patt_no"]=patt_no
                            dat["time"] = dat["time"] - dat["time"].min()
                            dat["Presentation"] = patt_no_count
                            dataS.append(dat)
                    
        pd.concat(dataS).to_csv("./testing_data/training_"+str(patt_no_count)+"_presentations.csv")
        
        
        
    def test_art(self, audi, arti,  patt_no_count, num_reps=10, t_on=16, t_off=30):
        ensure_directory_exists("./testing_data")
        ensure_directory_exists("./spikes")
        stim_strength = 500
        nest.SetKernelStatus({'overwrite_files': True})

        dataS = []
        spikes_tot  = []
        print("TESTING")
        for stim in range(0, 4):
            print("      STIM:   "+str(stim))
            for patt_no in range(0, len(audi)):
            
                run_twice = (stim == 0 and patt_no == 0)
                iterations = 2 if run_twice else 1
                
                for i in range(iterations):
                    with nest.RunManager():
                        nest.SetKernelStatus({'overwrite_files': True})
                        
                        stim_specs_test = stim_specs_patt_no_testing_arti_only(arti, patt_no, stim_strength)
                        
    #                    for i in range(0, 5):
    #                        stim_specs_pre_1 = stim_specs_pre(50)
    #                        self.stimulation_on(stim_specs_pre_1)
    #                        nest.Run(1)
    #                        self.stimulation_off()
    #                        nest.Run(2)
                            
                        #self.stimulation_off()
                        #nest.Run(2)
                        
                        print("patt_no: "+str(patt_no))
                        stim_specs_test = stim_specs_patt_no_testing_arti_only(arti, patt_no, stim_strength)
                        #stim_specs_test = stim_specs_patt_no(patt_no, nb_pattern, audi, audi, audi, audi,stim_strength)
            
                        self.stimulation_off()
                        nest.Run(5)
                        
                        self.stimulation_on(stim_specs_test)
                        nest.Run(t_on)
                        
                        counter_stim_pause = 0
                        self.stimulation_off()
                        
                        gi_AB = self.areas["AB"].glob.get(output="pandas")["V_m"].values[0]
                        gi_PM_i = self.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]
                        gi_PB = self.areas["PB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        gi_PF_i = self.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        
                        print("gi_AB:", gi_AB)
                        print("gi_PM_i:", gi_PM_i)
                        #while ((gi_AB > 0.75) | (gi_PM_i >0.75) | (counter_stim_pause < 60)):
                        while ((gi_PB > 0.7) | (gi_PF_i >0.7) | (counter_stim_pause < t_off)):
                            # Print values before update


                            nest.Run(0.5)
                            gi_AB = self.areas["AB"].glob.get(output="pandas")["V_m"].values[0]
                            gi_PM_i = self.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]
                            gi_PB = self.areas["PB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                            gi_PF_i = self.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
    #                        for area_eph in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L','A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']:
    #                            spikes_eph = np.sum(f.areas[area_eph].exc.get("phi", output="pandas")["phi"].values)
    #                            spikes_tot.append([counter_stim_pause, area_eph,  spikes_eph, stim,patt_no])
    #
                            
                            counter_stim_pause += 0.5

                        # Print values after update
                        print("gi_AB:", gi_AB)
                        print("gi_PM_i:", gi_PM_i)
                        print("counter_stim_pause:", counter_stim_pause)
                        print("#######################")
            
            
                        if iterations==1:
                            dat = dat_from_file('felix-*.dat')
                            dat['sum'] = dat['matrix'].apply(sum_arrays)
                            dat["stim"] = stim
                            dat["patt_no"]=patt_no
                            dat["time"] = dat["time"] - dat["time"].min()
                            dat["Presentation"] = patt_no_count
                            dataS.append(dat)
                    
        pd.concat(dataS).to_csv("./testing_data/arti_training_"+str(patt_no_count)+"_presentations.csv")


    def test_vis_motor(self, visual, motor,  patt_no_count, num_reps=10, t_on=16, t_off=30):
        ensure_directory_exists("./testing_data")
        ensure_directory_exists("./spikes")
        stim_strength = 500
        nest.SetKernelStatus({'overwrite_files': True})

        dataS = []
        spikes_tot  = []
        print("TESTING")
        for stim in range(0, 4):
            print("      STIM:   "+str(stim))
            for patt_no in range(0, len(visual)):
            
                run_twice = (stim == 0 and patt_no == 0)
                iterations = 2 if run_twice else 1
                
                for i in range(iterations):
                    with nest.RunManager():
                        nest.SetKernelStatus({'overwrite_files': True})
                        

                        print("patt_no: "+str(patt_no))
                        stim_specs_test = stim_specs_patt_no_testing_visual_motor_only(visual, motor, patt_no, stim_strength)
                        #stim_specs_test = stim_specs_patt_no(patt_no, nb_pattern, audi, audi, audi, audi,stim_strength)
            
                        self.stimulation_off()
                        nest.Run(5)
                        
                        self.stimulation_on(stim_specs_test)
                        nest.Run(t_on)
                        
                        counter_stim_pause = 0
                        self.stimulation_off()
                        
                        gi_AB = self.areas["AB"].glob.get(output="pandas")["V_m"].values[0]
                        gi_PM_i = self.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]
                        gi_PB = self.areas["PB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        gi_PF_i = self.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        
                        print("gi_AB:", gi_AB)
                        print("gi_PM_i:", gi_PM_i)
                        #while ((gi_AB > 0.75) | (gi_PM_i >0.75) | (counter_stim_pause < 60)):
                        while ((gi_PB > 0.7) | (gi_PF_i >0.7) | (counter_stim_pause < t_off)):
                            # Print values before update


                            nest.Run(0.5)
                            gi_AB = self.areas["AB"].glob.get(output="pandas")["V_m"].values[0]
                            gi_PM_i = self.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]
                            gi_PB = self.areas["PB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                            gi_PF_i = self.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
    #                        for area_eph in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L','A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']:
    #                            spikes_eph = np.sum(f.areas[area_eph].exc.get("phi", output="pandas")["phi"].values)
    #                            spikes_tot.append([counter_stim_pause, area_eph,  spikes_eph, stim,patt_no])
    #
                            
                            counter_stim_pause += 0.5

                        # Print values after update
                        print("gi_AB:", gi_AB)
                        print("gi_PM_i:", gi_PM_i)
                        print("counter_stim_pause:", counter_stim_pause)
                        print("#######################")
            
            
                        if iterations==1:
                            dat = dat_from_file('felix-*.dat')
                            dat['sum'] = dat['matrix'].apply(sum_arrays)
                            dat["stim"] = stim
                            dat["patt_no"]=patt_no
                            dat["time"] = dat["time"] - dat["time"].min()
                            dat["Presentation"] = patt_no_count
                            dataS.append(dat)
                    
        pd.concat(dataS).to_csv("./testing_data/visual_motor_training_"+str(patt_no_count)+"_presentations.csv")


    def test_aud_and_arti(self, audi, arti,  patt_no_count, num_reps=10, t_on=16, t_off=84):
        ensure_directory_exists("./testing_data")
        stim_strength = 500
        nest.SetKernelStatus({'overwrite_files': True})

        dataS = []
        print("TESTING")
        for stim in range(0, 2):
            print("      STIM:   "+str(stim))
            for patt_no in range(0, len(audi)):
                with nest.RunManager():
                    nest.SetKernelStatus({'overwrite_files': True})
                    print("patt_no: "+str(patt_no))
                    stim_specs_test = stim_specs_patt_no_testing_audi_and_artionly(audi, arti, patt_no, stim_strength)
                    #stim_specs_test = stim_specs_patt_no(patt_no, nb_pattern, audi, audi, audi, audi,stim_strength)
        
                    
                    self.stimulation_on(stim_specs_test)
                    nest.Run(t_on)
                    self.stimulation_off()
                    nest.Run(t_off)
                    
                    
                    dat = dat_from_file('felix-*.dat')
                    dat['sum'] = dat['matrix'].apply(sum_arrays)
                    dat["patt_no"]=patt_no
                    dat["stim"] = stim
                    dat["time"] = dat["time"] - dat["time"].min()
                    dataS.append(dat)
        pd.concat(dataS).to_csv("./testing_data/training_audi_and_arti"+str(patt_no_count)+"_presentations.csv")
                        
                        
                        
class Restore_Area:
    
     def __init__(self, directory_network, area):
     

        with open(directory_network, "rb") as f:
            network = pickle.load(f)

        excitatory_neurons = network["excitatory_neurons"]
        inhibitory_neurons = network["inhibitory_neurons"]
        global_inhibition = network["global_inhibition"]
        #pg_neuron = network["poisson_noise"]


        excitatory_neurons_area = excitatory_neurons[excitatory_neurons.area==area]
        inhibitory_neurons_area = inhibitory_neurons[inhibitory_neurons.area==area]
        global_inhibition_area = global_inhibition[global_inhibition.area==area]

        ########################################
        #   Recreate area with right parameters
        ########################################
        ## Create Excitatory ==> need to improve retrieval param
        self.exc = nest.Create("felix_exc",  n = excitatory_neurons_area.global_id.nunique(),
                                        params={'om': excitatory_neurons_area.om.unique()[0],
                                                'k_1': excitatory_neurons_area.k_1.unique()[0],
                                                'alpha': excitatory_neurons_area.alpha.unique()[0],
                                                'alpha_e': excitatory_neurons_area.alpha_e.unique()[0],
                                                'tau_adapt':excitatory_neurons_area.tau_adapt.unique()[0],
                                                'tau_m':excitatory_neurons_area.tau_m.unique()[0],
                                                'k_2': excitatory_neurons_area.k_2.unique()[0],
                                                'Jexcitatory':network["excitatory_neurons"].Jexcitatory.unique()[0]})#, positions=pos)


        ## Create Inhibitory ==> need to improve retrieval param
        self.inh = nest.Create("felix_inh", n = inhibitory_neurons_area.global_id.nunique(),
                                          params={'k_1': inhibitory_neurons_area.k_1.unique()[0],
                                                  'tau_m': inhibitory_neurons_area.tau_m.unique()[0]})


        ## Create Global ==> need to improve retrieval param
        self.glob = nest.Create("felix_inh", n= 1,
                                          params={'k_1':global_inhibition.k_1.unique()[0],
                                                  'tau_m': global_inhibition.tau_m.unique()[0]})
                                                  
        external_areas  = ["A1","V1","M1_L","M1_i"]
        
#        external_areas = ['V1',
#                                     'TO',
#                                     'AT',
#                                     'PF_L',
#                                     'PM_L',
#                                     'M1_L',
#                                     'A1',
#                                     'AB',
#                                     'PB',
#                                     'PF_i',
#                                     'PM_i',
#                                     'M1_i']
        if area in external_areas:
            print("Adding pg noise to: ", area)
            self.pg = nest.Create("poisson_generator", params = {"rate":10}) # before 20
            nest.Connect(self.pg, self.exc, syn_spec={"synapse_model":"static_synapse",
                                                      "receptor_type":3, "weight":1,
                                              "delay":0.5})


        connection_type = "static_synapse"
        #connection_type = "abs_synapse"
         

        global_id_area_exc = excitatory_neurons_area.global_id.unique().tolist()
        global_id_area_inh = inhibitory_neurons_area.global_id.unique().tolist()
        global_id_area_gi = global_inhibition.global_id.unique().tolist()

        weight_area_exc = network["weight"][(network["weight"].source.isin(global_id_area_exc))&(network["weight"].target.isin(global_id_area_exc))]
        weight_area_exc_inh = network["weight"][(network["weight"].source.isin(global_id_area_exc))&(network["weight"].target.isin(global_id_area_inh))]
        weight_area_inh_exc = network["weight"][(network["weight"].target.isin(global_id_area_exc))&(network["weight"].source.isin(global_id_area_inh))]

        weight_area_exc_gi = network["weight"][(network["weight"].source.isin(global_id_area_exc))&(network["weight"].target.isin(global_id_area_gi))]
        weight_area_gi_exc = network["weight"][(network["weight"].source.isin(global_id_area_gi))&(network["weight"].target.isin(global_id_area_exc))]




        self.e_e_syn = nest.Connect(weight_area_exc["source"].values, weight_area_exc["target"].values,
                             "one_to_one",
                             {"synapse_model":connection_type,
                              "weight": weight_area_exc["weight"].values,
                             "delay":weight_area_exc["delay"].values,
                              'receptor_type': 1
                             })#,
                            #return_synapsecollection=True)
         

        #nest.Connect(network["e_syns"].source.values, network["e_syns"].target.values,
                            # "one_to_one",
                            # {"synapse_model": "e_syn", "weight": network["e_syns"].weight.values})
         

        # Exc -> Inh -> Exc connections
        nest.Connect(weight_area_exc_inh["source"].values,
                     weight_area_exc_inh["target"].values,
                     'one_to_one',
                     {'synapse_model': 'static_synapse',
                      'weight':weight_area_exc_inh["weight"].values,
                      'delay': weight_area_exc_inh["delay"].values})


        # Inh -> Exc connections
        nest.Connect(weight_area_inh_exc["source"].values,
                     weight_area_inh_exc["target"].values,
                     'one_to_one',
                     syn_spec={'synapse_model': 'static_synapse',
                    'receptor_type': 2,
                    'delay': weight_area_inh_exc["delay"].values,
                    'weight': weight_area_inh_exc["weight"].values})

        # global inhibition
        nest.Connect(weight_area_exc_gi["source"],
                     weight_area_exc_gi["target"],
                     'one_to_one',
                     syn_spec={
                    'synapse_model':'static_synapse',
                'delay': weight_area_exc_gi["delay"].values,
                'weight': weight_area_exc_gi["weight"].values})

        nest.Connect(weight_area_gi_exc["source"],
                     weight_area_gi_exc["target"],
                     'one_to_one',
                     syn_spec={
                'receptor_type': 2,
                'delay':weight_area_gi_exc["delay"].values,
                'weight':weight_area_gi_exc["weight"].values})


        self.exc_stim = None


     def stimulation_on(self, neurons, I_stim):
        """
        Turn on stimulation for set ot neurons.
    
        neurons - list of indices of neurons to stimulate within area (0..624)
        I_stim - stimulation current to set on neurons
        """
    
        self.exc_stim = self.exc[neurons]
        self.exc_stim.I_e = I_stim
    
     def stimulation_off(self):
        """
        Turn stimulation off again for stimulated neurons.
        """
    
        if self.exc_stim:
            self.exc_stim.I_e = 0
            self.exc_stim = None
            
            






def testing_auditory_multiple_networks(networks_list, networks_dir, network_out):
    ensure_directory_exists(network_out)
    
    for patt_no_count in networks_list:
        print("##############################")
        print("     NETWORK:  "+str(patt_no_count))
        print("##############################")

        filename = "network_"+str(patt_no_count)
        directory = networks_dir

        directory_network = directory+filename
        with open(directory_network, "rb") as f:
            network = pickle.load(f)
            
        audi = network["pattern_auditory"]
        arti = network["pattern_articulatory"]
        motor = network["pattern_motor"]
        visu = network["pattern_visual"]
        nb_pattern = len(audi)
        
        f = FelixNet()
        f.rebuild_net(directory_network)
        
        print("NETWORK REBUILT")
        
        f.test_aud(audi, audi,  patt_no_count, num_reps=10, t_on=2, t_off=30)
        
        df = pd.read_csv("./testing_data/training_"+str(patt_no_count)+"_presentations.csv")
        
        add_rows =  []

        for stim in df.stim.unique():
            for area in df.AreaAbs.unique():
                for patt_no in df.patt_no.unique():
                    for time in df.time.unique():
                        eph = df[(df.AreaAbs==area)&(df.patt_no==patt_no)&(df.time==time)&(df.stim==stim)]
            
                        if len(eph)==0:
                            add_rows.append([area, patt_no, time, stim, 0])


        add_rows = pd.DataFrame(add_rows, columns=["AreaAbs", "patt_no", "time", "stim","sum"])
        add_rows
        df = pd.concat([df, add_rows])
        df["Presentation"]=df["Presentation"].ffill()
        df["Cond"]="Audi"
        df = df[df.time<40]
        df.to_csv("./"+network_out+"/training_"+str(patt_no_count)+"_presentations_cleaned.csv")
        
        

        
    print("##############################")
    print("FINAL END")
    print("##############################")
    
    

    
    
def testing_articulatory_multiple_networks(networks_list, networks_dir, network_out):
    ensure_directory_exists(network_out)

    for patt_no_count in networks_list:
        print("##############################")
        print("     NETWORK:  "+str(patt_no_count))
        print("##############################")

        filename = "network_"+str(patt_no_count)
        directory = networks_dir

        directory_network = directory+filename
        with open(directory_network, "rb") as f:
            network = pickle.load(f)
            
        audi = network["pattern_auditory"]
        arti = network["pattern_articulatory"]
        motor = network["pattern_motor"]
        visu = network["pattern_visual"]
        nb_pattern = len(audi)
        
        f = FelixNet()
        f.rebuild_net(directory_network)
        
        f.test_art(arti, arti,  patt_no_count, num_reps=10, t_on=2, t_off=30)
        
        df = pd.read_csv("./testing_data/arti_training_"+str(patt_no_count)+"_presentations.csv")
        
        add_rows =  []

        for stim in df.stim.unique():
            for area in df.AreaAbs.unique():
                for patt_no in df.patt_no.unique():
                    for time in df.time.unique():
                        eph = df[(df.AreaAbs==area)&(df.patt_no==patt_no)&(df.time==time)&(df.stim==stim)]
            
                        if len(eph)==0:
                            add_rows.append([area, patt_no, time, stim, 0])


        add_rows = pd.DataFrame(add_rows, columns=["AreaAbs", "patt_no", "time", "stim","sum"])
        add_rows
        df = pd.concat([df, add_rows])
        df["Presentation"]=df["Presentation"].ffill()
        df = df[df.time<40]
        df["Cond"]="Arti"
        df.to_csv("./"+network_out+"/arti_training_"+str(patt_no_count)+"_presentations_cleaned.csv")
        
        
    
        
    print("##############################")
    print("FINAL END")
    print("##############################")


def testing_visual_motor_multiple_networks(networks_list, networks_dir, network_out):
    ensure_directory_exists(network_out)
    
    for patt_no_count in networks_list:
        print("##############################")
        print("     NETWORK:  "+str(patt_no_count))
        print("##############################")

        filename = "network_"+str(patt_no_count)
        directory = networks_dir

        directory_network = directory+filename
        with open(directory_network, "rb") as f:
            network = pickle.load(f)
            
        audi = network["pattern_auditory"]
        arti = network["pattern_articulatory"]
        motor = network["pattern_motor"]
        visu = network["pattern_visual"]
        nb_pattern = len(audi)
        
        f = FelixNet()
        f.rebuild_net(directory_network)
        
        print("NETWORK REBUILT")
        
        f.test_vis_motor(visu, motor,  patt_no_count, num_reps=10, t_on=2, t_off=30)
        
        df = pd.read_csv("./testing_data/visual_motor_training_"+str(patt_no_count)+"_presentations.csv")
        
        add_rows =  []

        for stim in df.stim.unique():
            for area in df.AreaAbs.unique():
                for patt_no in df.patt_no.unique():
                    for time in df.time.unique():
                        eph = df[(df.AreaAbs==area)&(df.patt_no==patt_no)&(df.time==time)&(df.stim==stim)]
            
                        if len(eph)==0:
                            add_rows.append([area, patt_no, time, stim, 0])


        add_rows = pd.DataFrame(add_rows, columns=["AreaAbs", "patt_no", "time", "stim","sum"])
        add_rows
        df = pd.concat([df, add_rows])
        df["Presentation"]=df["Presentation"].ffill()
        df["Cond"]="Visual_Motor"
        df = df[df.time<40]
        df.to_csv("./"+network_out+"/visual_motor_training_"+str(patt_no_count)+"_presentations_cleaned.csv")
        
        

        
    print("##############################")
    print("FINAL END")
    print("##############################")



networks = ["save_network"]


for network in networks:

    networks_list = [ 10]
    networks_dir = '/Users/maximecarriere/src/nest_test/training_action_object_refracto/'+network+"/"
    network_out = "P2_testing_test_"+network
    testing_visual_motor_multiple_networks(networks_list, networks_dir, network_out)
    testing_auditory_multiple_networks(networks_list, networks_dir, network_out)
    #testing_articulatory_multiple_networks(networks_list, networks_dir, network_out)

