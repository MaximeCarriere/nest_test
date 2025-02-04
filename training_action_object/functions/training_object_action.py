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
#from function_annexe import *








#
# Example Felix Network
#
# This example implements the network studied in
#
#    Tomasello R, Garagnani M, Wennekers T and Pulvermüller F (2018).
#    A Neurobiologically Constrained Cortex Model of Semantic Grounding
#    With Spiking Neurons and Brain-Like Connectivity.
#    Front. Comput. Neurosci. 12:88. 
#    DOI: https::10.3389/fncom.2018.00088
#
# Only some of the inter-area connections shown in Fig 1B are included.

class Area:
    """
    Single area of Felix model.

    For details, see felix_single_area.ipynb.
    """
    
    def __init__(self):
        
        pos = nest.spatial.grid(shape=[25, 25], extent=[25., 25.], edge_wrap=True)
        self.exc = nest.Create('felix_exc', params={'om': 0,
                                    'alpha': 0.0006,
                                     'alpha_e': 0.05,
                                    'tau_adapt':13., 
                                    'k_2': 0.8, ## noise
                                    'Jexcitatory':Jexcitatory}, positions=pos)
        self.inh = nest.Create('felix_inh',  params={'k_1': 0.04, #0.30/5
                                    'tau_m': 10.},
                                   positions=pos)
        self.glob = nest.Create('felix_inh', params={'k_1': 0.053*sJslow, #0.0066
                                     'tau_m': 15.0})

        # Exc -> Exc connections
        self.e_e_syn = nest.Connect(self.exc, self.exc,
                     {'rule': 'pairwise_bernoulli', 
                        'p': 0.15* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x, 
                        nest.spatial.distance.y, 
                        std_x=4.5,
                         std_y=4.5,
                         mean_x = 1,
                         mean_y= 1,
                         rho =0.70710678
                             ), 
                       'mask': {'grid': {'shape': [19, 19]}, 'anchor': [9, 9]}}, 
                     {'synapse_model': 'abs_synapse', 'receptor_type': 1, 
                      'weight': nest.random.uniform(e_e_min, e_e_max), 'delay': nest.resolution}, # orig: 'weight': nest.random.uniform(0, 0.1)
                      return_synapsecollection=True)

        # Exc -> Inh -> Exc connections
        nest.Connect(self.exc, self.inh, 'one_to_one', syn_spec={'synapse_model': 'static_synapse', 'delay': nest.resolution})
        
        # Inh -> Exc connections
        nest.Connect(self.inh, self.exc,
                     {'rule': 'pairwise_bernoulli', 'p': 1,
                      'mask': {'grid': {'shape': [5, 5]}, 'anchor': [2, 2]}},
                     syn_spec={'synapse_model': 'static_synapse', 'receptor_type': 2, 'delay': nest.resolution, 'weight': -1*sJinh})
        
        # global inhibition
        nest.Connect(self.exc, self.glob, syn_spec={'delay': nest.resolution, 'weight': 1})
        nest.Connect(self.glob, self.exc, syn_spec={'receptor_type': 2, 'weight':-1*sJslow})

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

        
class FelixNet:
    """
    Class representing entire network.
    """
    
    def __init__(self):

        try: 
            print("init")
            nest.ResetKernel()
            nest.set(resolution=1, local_num_threads=16, rng_seed=12345)
        except:
            print("none")

    def connect_areas(self):
        """
        Create inter-area connections.
        """

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
                         'weight': nest.random.uniform(between_min, between_max), 'delay': nest.resolution}) # orig 'weight': nest.random.uniform(0, 0.1)

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


    def train_action_object(self, motor, visu, audi, arti, num_reps=10, t_on=16, t_off=84, stim_specs=None, nb_pattern=2):

        ensure_directory_exists("./training_data")
        ensure_directory_exists("./plot_weight")
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
                    stim_specs = stim_specs_patt_no(patt_no, nb_pattern, motor, visu, audi, arti)
                
                    ## Count presentation
                    patt_no_count[patt_no] += 1
                
                    print("Presentation Count: "+str(patt_no_count))
                    
                
                    self.stimulation_on(stim_specs)
                    nest.Run(t_on)
                
                    self.stimulation_off()
                    nest.Run(t_off)

                

                if patt_no_count[-1]%20==0:
                    dat=dat_from_file('felix-*.dat')
                    dat['sum'] = dat['matrix'].apply(sum_arrays)
                    dat["Pres"] = patt_no_count[-1]
                    dat["patt_no"] = patt_no
                    dat.to_csv("./training_data/training_"+str(patt_no_count[-1])+"_presentations.csv")
                    save_plot_weight(patt_no_count[-1])
                    save_plot_activation(patt_no_count[-1], dat)
                    

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
                    





# GLOBAL VARIABLES for parameter changes

sJslow= 4.6 #4.6 before #single_area: 24. # orig:55, modified: 55./4, good vals: 20-25
sJinh= 23  #single area: 6.


Jexcitatory = 17.3
## 16.3 ==> works but no weight augmentation after 30 Presentations increase to 16.8
## 16.8 ==> Less activation? Why =+> inhibition proportional might require to remove a bit of inhibition
## Back to 16.3 with sJinh = 38 ==> still show some activation but no weight augmentation after 20 Presentations ==> reduce Inhi to 35
## 16.3 with sJinh = 35 same ==> No learning augmentation after 20 presentations and more than 34% of Weight are really low
## 16.3 with sJinh = 23

#######################################
# Concern about the weight definition
# I am not sure but I feel like when we define a maximum at the begining
# the weight won't be able to go over it
# Need more analysis
#######################################

## within-area
e_e_min=0.099 # orig:0.1
e_e_max= 0.1 #0.225 # orig:1
## between-area
between_min=0.099 # orig:0.1
between_max=0.1#0.225 # orig:1
## stimulation strength
stim_strength= 110 #single area: 175


