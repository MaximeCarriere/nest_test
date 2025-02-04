
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
from functions.function_annexe import *

#from functions.training_object_action import *



nest.set_verbosity("M_ERROR")
nest.Install('felixmodule')







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
    
    def __init__(self, name):
        self.name = name



        pos = nest.spatial.grid(shape=[25, 25], extent=[25., 25.], edge_wrap=True)
        self.exc = nest.Create('felix_exc', params={'om': 0,
                                    'k_1':k_1_exc,
                                    'tau_m':tau_m_exc,
                                    'alpha': alpha_exc,
                                    'alpha_e': 0.05,
                                    'tau_adapt':tau_adapt_exc,
                                    'k_2': k_2, ## noise
                                    'Jexcitatory':Jexcitatory}, positions=pos)

                                    
        self.inh = nest.Create('felix_inh',  params={'k_1': k1_inh, #0.30/5
                                    'tau_m': tau_m_inh},
                                    
                                   positions=pos)
        self.glob = nest.Create('felix_inh', params={'k_1': k1_glob, #0.053*sJslow, #0.0066
                                     'tau_m': tau_m_glob})


        external_areas = ["A1", "V1", "M1_L", "M1_i"]
        if self.name in external_areas:
            print("Adding noise to external area:", self.name)
            
            self.pg = nest.Create('poisson_generator', params={'rate': 10})#orig 0.5
            # noise -> Exc
            nest.Connect(self.pg, self.exc, syn_spec={'synapse_model': 'static_synapse',
                                                      'receptor_type': 3, 'weight':1,
                                                       'delay':0.5})

        #else:
        #    print("Adding noise to internal area:", self.name)
        #    self.pg = nest.Create('poisson_generator', params={'rate': 1})#orig 0.5
            # noise -> Exc
        #    nest.Connect(self.pg, self.exc, syn_spec={'synapse_model': 'static_synapse',
        #                                              'receptor_type': 3, 'weight':1,
        #                                               'delay':0.5})

            
                # Exc -> Exc connections
        self.e_e_syn = nest.Connect(self.exc, self.exc,
                     {'rule': 'pairwise_bernoulli',
                        'p': 0.15* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        nest.spatial.distance.y,
                        std_x=3.2,
                         std_y=3.2,
                         mean_x = 0,
                         mean_y= 0,
                         rho =0
                             ),
                       'mask': {'grid': {'shape': [19, 19]}, 'anchor': [9, 9]}},
                     {'synapse_model': 'abs_synapse', 'receptor_type': 1,
                      'weight': nest.random.uniform(e_e_min, e_e_max), 'delay': 1}, # orig: 'weight': nest.random.uniform(0, 0.1)
                      return_synapsecollection=True)



        # Exc -> Inh -> Exc connections
        nest.Connect(self.exc, self.inh,
                             {'rule': 'pairwise_bernoulli', 'p': 1,
                    'mask': {'grid': {'shape': [5, 5]}, 'anchor': [2, 2]}},
                     syn_spec={'synapse_model': 'static_synapse',  'delay': 0.5,
                     'weight': 0.295*nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        nest.spatial.distance.y,
                        std_x=1.42,
                         std_y=1.42,
                         mean_x = 0,
                         mean_y= 0,
                         rho =0
                             ),})

        #nest.Connect(self.exc, self.inh, 'one_to_one', syn_spec={'synapse_model': 'static_synapse',
        # 'delay': 0.5})

        # Inh -> Exc connections
        nest.Connect(self.inh, self.exc,
                 'one_to_one', syn_spec={'synapse_model': 'static_synapse', 'receptor_type': 2, 'delay': 0.5,
                 'weight':-1*sJinh})
        # global inhibition
        nest.Connect(self.exc, self.glob, syn_spec={'delay':0.5, 'weight': 1})
        nest.Connect(self.glob, self.exc, syn_spec={'delay':0.5, 'receptor_type': 2, 'weight':-1*sJslow})

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
            nest.set(resolution=0.5, local_num_threads=16, rng_seed=12)
        except:
            print("none")

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
            ('PF_i', 'PM_i'),
            ('PF_i', 'M1_i'),
            ('PM_i', 'M1_i'),
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

        jumping_link = [('V1', 'AT'),
                        ('PF_L', 'M1_L'),
                        ('A1', 'PB'),
                        ('PF_i', 'M1_i'),
                        ('TO', 'PF_L'),
                        ('AT', 'PM_L'),
                        ('AB', 'PF_i'),
                        ('PB', 'PM_i'),
                        ('AT', 'PF_L'),
                        ('AT', 'PB'),
                        ('AT', 'PF_i'),
                        ('PF_L', 'PB'),
                        ('PF_L', 'PF_i'),
                        ('PB', 'PF_i'),
        ]

        next_neighbourg_system = [('V1', 'TO'),
                                ('TO', 'AT'),
                                ('PF_L', 'PM_L'),
                                ('PM_L', 'M1_L'),
                                ('A1', 'AB'),
                                ('AB', 'PB'),
                                ('PF_i', 'PM_i'),
                                ('PM_i', 'M1_i')]


        cross_system = [
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
            ('PB', 'PM_i')]

        within_system = [
            ('V1', 'TO'),
            ('V1', 'AT'),
            ('TO', 'AT'),
            ('PF_L', 'PM_L'),
            ('PF_L', 'M1_L'),
            ('PM_L', 'M1_L'),
            ('A1', 'AB'),
            ('A1', 'PB'),
            ('AB', 'PB'),
            ('PF_i', 'PM_i'),
            ('PF_i', 'M1_i'),
            ('PM_i', 'M1_i')]
            
                    # if you do not want reciprocal area you should comment the next line
        #connectome.extend(list((tgt, src) for src, tgt in connectome))


        # Now create connections
        #for src, tgt in connectome:
        #    nest.Connect(self.areas[src].exc, self.areas[tgt].exc,
        #                {'rule': 'pairwise_bernoulli',
                        # 'p': 0.29* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        # #'p': 0.15* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        # nest.spatial.distance.y,
                        # std_x=4.5,
                        #  std_y=4.5,
                        #  mean_x = 0,
                        #  mean_y= 0,
                        #  rho =0
                        #      ),

        #                'p': 0.13* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        #'p': 0.15* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
        #                nest.spatial.distance.y,
        #                std_x=9,
        #                 std_y=9,
        #                 mean_x = 0,
        #                 mean_y= 0,
        #                 rho =0
        #                     ),
        #                  'mask': {'grid': {'shape': [19, 19]}, 'anchor': [9, 9]}},
        #                 {'synapse_model': 'abs_synapse', 'receptor_type': 1,
        #                 'weight': nest.random.uniform(between_min, between_max), 'delay': 1.5}) # orig 'weight': nest.random.uniform(0, 0.1)


                # if you do not want reciprocal area you should comment the next line
        cross_system.extend(list((tgt, src) for src, tgt in cross_system))


        # Now create connections
        for src, tgt in cross_system:
            nest.Connect(self.areas[src].exc, self.areas[tgt].exc,
                        {'rule': 'pairwise_bernoulli',
                        'p': 0.13*0.5* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        #'p': 0.15* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        nest.spatial.distance.y,
                        std_x=9,
                         std_y=9,
                         mean_x = 0,
                         mean_y= 0,
                         rho =0
                             ),
                          'mask': {'grid': {'shape': [19, 19]}, 'anchor': [9, 9]}},
                         {'synapse_model': 'abs_synapse', 'receptor_type': 1,
                         'weight': nest.random.uniform(between_min, between_max), 'delay': 1}) # orig 'weight': nest.random.uniform(0, 0.1)


                        # if you do not want reciprocal area you should comment the next line
        within_system.extend(list((tgt, src) for src, tgt in within_system))



        # Now create connections
        for src, tgt in within_system:
            nest.Connect(self.areas[src].exc, self.areas[tgt].exc,
                        {'rule': 'pairwise_bernoulli',
                        'p': 0.13* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        #'p': 0.15* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                        nest.spatial.distance.y,
                        std_x=9,
                         std_y=9,
                         mean_x = 0,
                         mean_y= 0,
                         rho =0
                             ),
                          'mask': {'grid': {'shape': [19, 19]}, 'anchor': [9, 9]}},
                         {'synapse_model': 'abs_synapse', 'receptor_type': 1,
                         'weight': nest.random.uniform(between_min, between_max), 'delay': 1}) # orig 'weight': nest.random.uniform(0, 0.1)



        

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
        
        #self.areas = {area: Area() for area in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L',
        #                                        'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']}
                                                
        self.areas = {area: Area(area) for area in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L',
                                                'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']}
#                                                
        self.connect_areas()
        self.connect_recorders()


    def store(self, filename, motor, visu, audi, arti):
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
        exc_neurons_param_list = ["om", "alpha", "alpha_e", "tau_adapt", "k_2", "Jexcitatory","tau_m"]
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
                
        
        directory = "./save_network/"

        print("SAVING NETWORK ==> ")
        ensure_directory_exists(directory)
        
        with open(directory+filename, "wb") as f:
                    pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)


    # def store(self, filename, motor, visu, audi, arti, chunksize=5000):
    #     """
    #     Store neuron membrane potential and synaptic weights to given file.

    #     Args:
    #         chunksize (int, optional): Size of each data chunk to write to the file. Defaults to 10000.
    #     """

    #     assert nest.NumProcesses() == 1, "Cannot dump MPI parallel"

    #     ## Inhibitory Parameters 
    #     inh_neurons_param_list = ["k_1", "tau_m"]
    #     list_param_value_inh = pd.DataFrame([[param, list(set(self.areas['A1'].inh.get([param], output="pandas")[param].values.tolist()))] for param in inh_neurons_param_list], columns=["param", "value"])

    #     ## Excitatory Parameters
    #     exc_neurons_param_list = ["om", "alpha", "alpha_e", "tau_adapt", "k_2", "Jexcitatory"]
    #     list_param_value = pd.DataFrame([[param, list(set(self.areas['A1'].exc.get([param], output="pandas")[param].values.tolist()))] for param in exc_neurons_param_list], columns=["param", "value"])

    #     ## Global Inhibition Parameters
    #     glob_neurons_param_list = ["k_1", "tau_m"]
    #     list_param_value_glob = pd.DataFrame([[param, list(set(self.areas['A1'].glob.get([param], output="pandas")[param].values.tolist()))] for param in glob_neurons_param_list], columns=["param", "value"])

    #     ## All weights
    #     test = nest.GetConnections().get(
    #                 ("delay", "receptor", "source", "synapse_model", "target", "weight"), output="pandas"
    #             )

    #     with open(f"./save_network/{filename}", "wb") as f:
    #         pickle.dump({"param_excitatory": list_param_value,
    #                     "param_inhibitory": list_param_value_inh,
    #                     "param_global": list_param_value_glob}, f, pickle.HIGHEST_PROTOCOL)

    #         # Iterate over areas and write neuron data in chunks
    #         for area in self.areas.keys():
    #             exc_data = self.areas[area].exc.get(output="pandas")
    #             num_chunks = (len(exc_data) // chunksize) + (1 if len(exc_data) % chunksize > 0 else 0)
    #             for i in range(num_chunks):
    #                 start_idx = i * chunksize
    #                 end_idx = min(start_idx + chunksize, len(exc_data))
    #                 chunk = exc_data.iloc[start_idx:end_idx]
    #                 pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)

    #             # Similar approach for inhibitory and global inhibition data
    #             inh_data = self.areas[area].inh.get(output="pandas")
    #             num_chunks = (len(inh_data) // chunksize) + (1 if len(inh_data) % chunksize > 0 else 0)
    #             for i in range(num_chunks):
    #                 start_idx = i * chunksize
    #                 end_idx = min(start_idx + chunksize, len(inh_data))
    #                 chunk = inh_data.iloc[start_idx:end_idx]
    #                 pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)

    #             # ... (similar loop for global inhibition data)

    #         # Finally, pickle the remaining data
    #         pickle.dump({"weight": test,
    #                     "pattern_motor": motor,
    #                     "pattern_visual": visu,
    #                     "pattern_auditory": audi,
    #                     "pattern_articulatory": arti}, f, pickle.HIGHEST_PROTOCOL)


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


    def train_action_object(self, motor, visu, audi, arti, num_reps=10, t_on=16, t_off=30, stim_specs=None, nb_pattern=2):

        ensure_directory_exists("./training_data")
        ensure_directory_exists("./plot_weight")
        ensure_directory_exists("./weight_data")
        ensure_directory_exists("./save_network")
        ensure_directory_exists("./gi")
        ensure_directory_exists("./li")
        #ensure_directory_exists("./training_nest")
        patt_no = 0
        
        f.store("network_start", motor, visu, audi, arti)
        gi_tot = []
        #li_tot = []
        patt_no_count = [0]*nb_pattern
        count_firs_pres = 0
        while any(count < num_reps for count in patt_no_count):
            with nest.RunManager():
                

                ## Choose patt_no randomly except for the 10 first presentation 
                if patt_no_count[-1]<0:
                    patt_no = patt_no +1
                    t_off = 300
                    if patt_no == nb_pattern:
                        patt_no = 0
                else:
                    ## Randomly pick a number
                    patt_no = random.randint(0, nb_pattern-1)
                    t_off = 30

                ## Make sure patt_no hasn't been presented too many times
                if patt_no_count[patt_no]>=num_reps:
                    continue

                else:
                    print("################")
                    print("Presentation patt_no: "+str(patt_no))
                    
                    nest.SetKernelStatus({'overwrite_files': True})
                
                    
                    ## Get the associated specificity (pattern)

                    stim_specs = stim_specs_patt_no(f, patt_no, nb_pattern, motor, visu, audi, arti, stim_strength)
                    
                    #if patt_no_count[patt_no]>100:
                    #        print("===>   STIM AUD ONLY   <===")
                    #        stim_specs =  stim_specs_patt_no_aud_only(f, patt_no, nb_pattern, motor, visu, audi, arti, stim_strength)
                    
                
                    ## Count presentation
                    patt_no_count[patt_no] += 1
                
                    print("Presentation Count: "+str(patt_no_count))
                    
                    #######################################################################################################
                                            # Make Sure no input is shown if GI too high when initalised
                                            # Correspond to the explosion of activation at the begining of the network
                    #######################################################################################################
                    if np.sum(patt_no_count)<2:
                        gi_AB = f.areas["AB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        gi_PM_i = f.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        #self.stimulation_on()
                        counter_stim_pause_init = 0
                        while ((gi_AB > 0.75) | (gi_PM_i >0.75) | (counter_stim_pause_init < 60)):
                            nest.Run(0.5)
                            gi_AB = f.areas["AB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                            gi_PM_i = f.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                            counter_stim_pause_init += 0.5
                            for area_eph in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L','A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']:
                                gi_eph = f.areas[area_eph].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                                #li_eph = np.sum(f.areas[area_eph].inh.get(output="pandas")["V_m"].values)
                                gi_tot.append([counter_stim_pause_init,area_eph,gi_eph, "stim_off", 999999])
                                #li_tot.append([counter_stim_pause_init,area_eph,li_eph, "stim_off"])
                        #print("counter_stim_pause_init:", counter_stim_pause_init)
                    #######################################################################################################

                    #print("#####################################################################################")
                    #print("#####################################################################################")
                    gi_AB = f.areas["AB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                    gi_PM_i = f.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                    
                    #print("GI_AB: "+str(gi_AB))
                    #print("GI_PM_i: "+str(gi_PM_i))
                    #print("#####################################################################################")
                    #print("#####################################################################################")
                    
                    self.stimulation_on(stim_specs)
                    #nest.Run(t_on)
                    counter_run = 0
                    for time_to_run in range(0, t_on):
                        counter_run = counter_run + 0.5
                        nest.Run(0.5)
                        for area_eph in ['A1', 'AB','PB','PF_i','PM_i','M1_i']:
                            gi_eph = f.areas[area_eph].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                            #li_eph = np.sum(f.areas[area_eph].inh.get(output="pandas")["V_m"].values)
                            gi_tot.append([count_firs_pres+counter_run+counter_stim_pause_init, area_eph,  gi_eph, "stim_on", counter_run, np.sum(patt_no_count)])
                            #li_tot.append([count_firs_pres+counter_run+counter_stim_pause_init, area_eph,  li_eph, "stim_on", counter_run, np.sum(patt_no_count)])
                        
                        

                        

                    counter_stim_pause = 0
                    self.stimulation_off()

                    
                    # Print initial values
                    #gi_AB = f.areas["AB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                    #gi_PM_i = f.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                    gi_PB = f.areas["PB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                    gi_PF_i = f.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow


                    #while ((gi_AB > 0.065) | (gi_PM_i >0.065) | (counter_stim_pause < t_off)):
                    while ((gi_PB > 0.70) | (gi_PF_i > 0.70) | (counter_stim_pause < t_off)):
                        # Print values before update


                        nest.Run(0.5)
                        #gi_AB = f.areas["AB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        #gi_PM_i = f.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        gi_PB = f.areas["PB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        gi_PF_i = f.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                        counter_stim_pause += 0.5
                        for area_eph in ['A1', 'AB','PB','PF_i','PM_i','M1_i']:
                            gi_eph = f.areas[area_eph].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                            #li_eph = np.sum(f.areas[area_eph].inh.get(output="pandas")["V_m"].values)
                            gi_tot.append([counter_stim_pause_init+count_firs_pres+counter_run+counter_stim_pause,area_eph,gi_eph, "stim_off",999999, np.sum(patt_no_count)])
                            #li_tot.append([counter_stim_pause_init+count_firs_pres+counter_run+counter_stim_pause,area_eph,li_eph, "stim_off",999999, np.sum(patt_no_count)])
                    #print("gi_AB: "+str(gi_AB))
                    #print("gi_PM_i: "+str(gi_PM_i))
                    #print("gi_PB: "+str(gi_PB))
                    #print("gi_PF_i: "+str(gi_PF_i))



                    #print("counter_stim_pause:", counter_stim_pause)
                    #pd.DataFrame(gi_tot, columns=["stp","area","GI","status","new_stp", "cond"]).to_csv("./gi/gi_pres_"+str(np.sum(patt_no_count))+"_presentations.csv")
                    
                    


                    count_firs_pres = count_firs_pres+counter_run+counter_stim_pause


                

                if np.sum(patt_no_count)%30==0:
                     dat=dat_from_file('felix-*.dat')
                     dat['sum'] = dat['matrix'].apply(sum_arrays)
                     dat["Pres"] = patt_no_count[patt_no]
                     dat["patt_no"] = patt_no
                #     #dat.to_csv("./training_data/training_"+str(patt_no_count[-1])+"_presentations.csv")
                     #save_plot_weight(f, patt_no_count[-1])
                     save_plot_activation_new(patt_no_count[patt_no], dat, patt_no)
                    #save_plot_activation_new(patt_no_count[patt_no], dat, patt_no)

                #if (patt_no_count[-1] in [1, 10, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000])&(patt_no==nb_pattern-1):
                #if (patt_no_count[-1] in [5,10, 50, 100, 200, 500])&(patt_no==nb_pattern-1):
                if (patt_no_count[-1] in [1, 10, 30, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1700, 2000])&(patt_no==nb_pattern-1):
                    f.store("network_"+str(patt_no_count[-1]), motor, visu, audi, arti)
                    

                #elif count_firs_pres==0:
                #    dat=dat_from_file('felix-*.dat')
                #    dat['sum'] = dat['matrix'].apply(sum_arrays)
                #    dat["Pres"] = 0
                #    dat["patt_no"] = patt_no
                #    dat.to_csv("./training_data/training_start.csv")  
                #    save_plot_weight(patt_no_count[-1])
                #s    save_plot_activation(patt_no_count[-1], dat)
                    
                #    count_firs_pres = 1


    
                #dat=dat_from_file('felix-*.dat')
                #dat['sum'] = dat['matrix'].apply(sum_arrays)
                #dat["Pres"] = patt_no_count[-1]
                #dat["patt_no"] = patt_no
                #dat.to_csv("./training_data/training_"+str(np.sum(patt_no_count))+"_presentations.csv")
                #save_plot_activation_new(patt_no_count[patt_no], dat, patt_no)



        dat['sum'] = dat['matrix'].apply(sum_arrays)
        dat["Pres"] = num_reps
        #dat.to_csv("./training_data/training_end.csv")
        save_plot_weight(f, patt_no_count[-1])
        #save_plot_activation(f, patt_no_count[-1], dat)
        save_plot_activation_new(patt_no_count[-1], dat, patt_no)
                    

#def create_act_obj_pattern(nb_pattern):
#    motor = []
#    visu = []
#    audi = []
#    arti = []
#    for i in range(0, nb_pattern):
#        motor.append(f.neurons2IDs(sorted(random.sample(list(range(0,625)),19))))
#        visu.append(f.neurons2IDs(sorted(random.sample(list(range(0,625)),19))))
#        audi.append(f.neurons2IDs(sorted(random.sample(list(range(0,625)),19))))
#        arti.append(f.neurons2IDs(sorted(random.sample(list(range(0,625)),19))))
#
#
#    return motor, visu, audi, arti


import random

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

    for i in range(0, nb_pattern):
        motor.append(f.neurons2IDs(sorted(random.sample(list(neuron_pool_motor), size_pattern))))
        visu.append(f.neurons2IDs(sorted(random.sample(list(neuron_pool_visu), size_pattern))))
        audi.append(f.neurons2IDs(sorted(random.sample(list(neuron_pool_audi), size_pattern))))
        arti.append(f.neurons2IDs(sorted(random.sample(list(neuron_pool_arti), size_pattern))))

        neuron_pool_motor -= set(motor[-1])
        neuron_pool_visu -= set(visu[-1])
        neuron_pool_audi -= set(audi[-1])
        neuron_pool_arti -= set(arti[-1])

    show_owerlapp_pattern(motor, visu, audi, arti)
    return motor, visu, audi, arti




# GLOBAL VARIABLES for parameter changes

sJslow= 65##6.3#3.3# 3.3 #4.6 before #single_area: 24. # orig:55, modified: 55./4, good vals: 20-25
sJinh= 500 #single area: 6
Jexcitatory = 500 #origin 500
k_2 = 50#.0005#0.5 # ==> Need to increase a biratt the noise

## within-area
e_e_min=0.01 # orig:0.1
e_e_max= 0.1 #0.225 # orig:1
## between-area
between_min=0.01 # orig:0.1
between_max=0.1#0.225 # orig:1
## stimulation strength
stim_strength = 500 #single area: 175


##  Excitatory Variable
k_1_exc = 0.01
tau_m_exc =  5#2.5#2.5 origin
alpha_exc = 0.01
tau_adapt_exc= 20 #origin 10

## Inhibitory Variable
k1_inh = 1
tau_m_inh =  10 #5 origin

## Global inhibition
k1_glob = 1
tau_m_glob = 24 #12 origin




## (1) sJslow = 4.6 and k_2 = 0.8 ==> works but too much excitation when reaching central areas
## (2) sJslow = 5.5 and K_2 0.8 ==> works but weight shrinks fast and they won't reach central areas after a couple of training
## (3) sJslow = 5.5 and K_2 0.0 ==> works but weight shrinks fast and they won't reach central areas after a couple of training (noise useless so far, let's do without them)
## (4) sJslow = 5.0 and K_2 = 0 ==> works but weight shrinks fast and they won't reach central areas after a couple of training (trained until 200 only 1 activated neurons in tedsting in non A1 areas)
## (5) sJslow = 4.8 and K_2 = 0 ==> too much excitation, needs more inhibition (trained until 130 presentations)
## (6) sJslow = 4.9 and K_2 = 0 ==> too much excitation, needs reduce excitation (trained until 50 presentations)
## (7) sJslow = 4.9 and K_2 = 0 Jexcitatory = 15.5 ==> stable but weight shrinks slowly, no activation in central area after 200 pres (testing) ==> increase slightly excitation
# (8) sJslow = 4.9 and K_2 = 0 Jexcitatory = 15.9 ==> after 500 presentations some CA takes over (6/12) over 600 neurons activation (at 200 pres the network behva normally with slight activation in central area 2-3 neurons)
# (8 bis) ==> the network learn slowly and seem to be almost right, but need to increase competition between CA (possibly increasing noise)
# (8 bis 2) ==> as the network should work with noise = 0 we will increase slightly excitation to provide more competition
# (9) sJslow = 4.9 and K_2 = 0 Jexcitatory = 16 ==> at 200 presentations it show low activation in central areas, after 500 merging occurs where all pattern trigger 600+ activation
# (9 bis) ==> it seems that our felix has way more weights at the begining (need more analysis), the excitation provided by one weight is so far too high providing mergining 
# (9 bis 2) ==> we should increase the number of weights and divide the excitatory parameters
## (10) sJslow = 4.9 and K_2 = 0 Jexcitatory = 8 increase within/betwen x 2 ==> problem when saving network (not possible to test)
# (10 bis) ==> there are some activation in central area but weight in A1 seem stable after 100 presentations, need a bit more excitation of increase number of weight
## (11) sJslow = 4.9 and K_2 = 0 Jexcitatory = 7 increase within/betwen x 3
# (11 bis) ==> after 50 presentations, one pattern trigger 600+ neurons the other are limited between 0 and 6
# (11 bis 2) ==> the weights increase quite quickly toward 0.2 


## Weird thing, same number of weight between A1 and PB ==> recording e_e_syn which are only within area connections
# ==> This is now correected with adjusted probability for within and between as well as initial weight value
# ==> They now match perfectly (+/- 1%) the number of connections in Felix

## NEW TESTS ==> LI same as Jexcitatory and GI = 4.9
# (1) Jexcitatory = 15 and k_2 = 0 after 70 presentations at least one patter trigger 1 neuron in a non stimulated areas ==> need to increase the excitation a little
# (2) Jexcitatory = 16 and k_2 = 0 same analysis as above, slight improvement ==> need to increase a bit the excitation (70 pres pattern with 1 neurons non stimulated areas)
# (3) Jexcitatory = 17 and k_2 = 0 ==> Same analysis as above, need to increase again excitatory
# (4) Jexcitatory = 19 and k_2 = 0 ==> same stop at 40 pres
# (5) Jexcitatory = 23 and k_2 = 0 ==> getting better but still needs some excitation (afet 170 presentations does not seem to have any change)
# (6) Jexcitatory = 25 and k_2 = 0 ==> still too low need to increase (70 pres)
# (7) Jexcitatory = 30 and k_2 = 0 ==> activation get better but no change after 100 pres (500 presentation in total)
# (8) Jexcitatory = 35 and k_2 = 0 ==> it seems to work fine (no merging, 8*-10 neurons activated in central areas after 1000 presentations)
# (8 bis) ==> the storing function was not useful, we will try to increase a bit the excitation and come back to this parameters if not working
# (9) Jexcitatory = 40 and k_2 = 0 ==> works well but not much learning happening after 500 presentations
# (9 bis) ==> when testing auditory for 500 and 1000 there are activation in the auditory system (6 neurons)
# (10) Jexcitatory = 60 and k_2 = 0 ==> 600+ activation after 30 pres
# (11) Jexcitatory = 50 and k_2 = 0 ==> the CA seems to works, high activation in central areas, no real mergining even after 1000 presentations, one or two pattern trigger huge activation +200
# (11 bis) ==> at 3000 there are some merging 
# (11 bis 2) ==> when doing a training again we can see that it seems to activate correctly the CA after 1000 presentations (semantic speificity, higher number in central areas)
# (11 bis 3) ==> However, when auditory testing only there is a weird effect with no reaching extra-sylvian region and high activation in articulatory


## While CA seems to form, there are still a bit of mergining and not a correct linkage between extra/pery sylvian region
# First some noise + higher global inhibition should resolve mergining 
# Second increase the Excitatory input should overcome the linkage between peri/extra ==> actutally I forgot to do reciprocal connections when loading network.
# ==> It should work better with this now

# NEw Test 
# (1) Jexcitatory = 50 and k_2 = 5 sJslow = 6.5 ==> no mergining, during training the CA seems to work but when testing there is some weird results
# (1 bis) ==> first the saving/loading network might not work properly cause instead of auditory input if we send motor input in A1 it seems to trigger more activate, is that normal?
# (1 bis 2) ==> second the A1 does not trigger activation in secondary area of extra-sylvian region, but does a little bit for primary
# (1 bis 3) ==> the good news is that the activation in primary extra is semantic ! With action in motor and Object in visual 
# (1 bis 4) ==> It might need to reduce the global inhibition that is too strong and avoid the activation of the secondary extra during training

####################### NOTE TO MYSELF #########################
# At the moment we do not have any noise in the inputs pattern, maybe that could help
# during training in felix there are some activation in primary areas from this noise as well
# Also pattern presentation show activation in primary areas during the first 5-6 stp before being shut down by GI
# Should we try to match exactly that? ==> at the moment it seems to work

# (1) Jexcitatory = 50 and k_2 = 5 sJslow = 5.5 ==> I stop after 500 presentations, it seems to work, but auditory presentation does not trigger more than auditory system ==> need to increase excitatory
# (2) Jexcitatory = 60 and k_2 = 5 sJslow = 5.5 ==> stop after 30 presentation ==> too much excitation always 600+ ==> reduce excitation
# (3) Jexcitatory = 55 and k_2 = 5 sJslow = 5.5 ==> stop at 1000, at 1000 one CA is taking over (100+), after 500 it looks nice with some activation in central extra-sylvian (semantic category) for aud testing
# (3 bis) ==> maybe the noise is too high, cause testing with no noise provides smaller activation (17 with, 8 without in auditory system)
# (4) Jexcitatory = 55 and k_2 = 1 sJslow = 5.5 ==> stop at 500, some merginin, let's try to reduce back to 50 the exctitation
# (5) Jexcitatory = 50 and k_2 = 1 sJslow = 5.5 ==> works well with semantic activation in extra-sylvian until primary areas
# (5 bis) ==> However the activaiton in extra-sylvian region after 3000 is quite low (max 3 neurons in central) ==> need to increase a bit the excitation or reduce noise?
# (6) Jexcitatory = 51 and k_2 = 1 sJslow = 5.5 ==> after 3000 on CA a bit too high (120+), the rest looks good with activation quite low in central extra (5 max)
# (6 bis) ==> maybe go back to (1) with way lower noise
# (7) Jexcitatory = 50 and k_2 = 0.05 sJslow = 5.5 ==> stop after 2000 one CA is taking over, the rest looks good
# (8) Jexcitatory = 50 and k_2 = 0.05 sJslow = 6.5 ==> Some pattern with too much activation after 300 (200+) and other that does not activate more than the auditory system
# (9) Jexcitatory = 50 and k_2 = 0.1 sJslow = 6 ==> stop at 1000 on CA is already at 70+
# (10) Jexcitatory = 50 and k_2 = 0.5 sJslow = 6 ==> stop after 1000, some CA triggers activation in extra-sylvian but most of them are stuck in auditoyr system


## NEW strategy, training until 3000 is too long we will now train until 500, try to find parameters with CA roughly the same size and activation in extra-sylvian
# In a later stage we can adjust parameters to avoid possible mergining after 500
# (1) Jexcitatory = 51 and k_2 = 0.5 sJslow = 5.6 ==> after 500 activation in central areas but quite small (extra) 
# (1 bis) ==> it seems that secondary extra-sylvian do not really process information for some reason, are they trimmed too much at first? not enough excitation for similar activation?
#(2) Jexcitatory = 52 and k_2 = 0.5 sJslow = 5.6 ==> after 500 presentations, it only activates the pery-sylvian regions and some only the auditory system
# (3) Jexcitatory = 60 and k_2 = 0.5 sJslow = 6.5 ==> after 30 presentations, the weird thing is semantic areas that received uncorrelated input seems to be activated way later 
# (3 bis) this uncorrelated areas seems not to received any inputs? or this prunning is really efficient?
# (3 bis 2) ==> at 200 presentations the excitation is huge (80 +), but the GI still put it below 

## It seems that in Felix the central areas are slightly activated by the inputs in the first place, and require the activation of the secondary areas to be fully activated
# In our current set of parameters central areas are directly activated, we therefore need to reduce the activation to have this steps of activations??
# Require more analysis with Felix
# ON GOING (1) Jexcitatory = 45 and k_2 = 0.5 sJslow = 6.5 ==> after 500 auditory stim only activate auditory system 

## NEW STRATEGY make it work perfectly for 10 trainign step then 50 then 500 and so on
## In Felix after 10 training presentation the activation is quite high in peri/extra with 40 sometimes 50 neurons in central areas
# Then at 50 presentations it goes down with secondary area are at 20 while central goes more to 15
# New set of parameters ==> goal at least 4 stp of activation in the stimulated area so sJslow around 5.5
# (1) Jexcitatory = 60 and sJslow = 5.5 works with same shape of activation but after 50 presentations activation restricted to auditory (a1 testing)
# (1 bis) after 100 presentations the excitation is better in auditory (15 neurons), but activation still restricted to Auditory
# (2) Jexcitatory = 60 and sJslow = 5 the global inhibition shut down directly the activation afer 2 stp in central areas even after 100 pres
# (2 bis) But the excitation is becoming a little too high, lets's reduce the tauslow from 15 to 12 (as in felix)
# (4) Jexcitatory = 50 and sJslow = 5.3 tauslow = 12 ==> excitation a little bit too low 
# ON GOIGN (3) Jexcitatory = 60 and sJslow = 5.3 tauslow = 12  k_2 = 0.05 ==>



# sJslow= 3.3 #4.6 before #single_area: 24. # orig:55, modified: 55./4, good vals: 20-25
# sJinh= 60  #single area: 6
# Jexcitatory = 60
# k_2 = 0.5 # ==> Need to increase a bit the noise
# tauslow = 8

# ## within-area
# e_e_min=0.0001 # orig:0.1
# e_e_max= 0.07 #0.225 # orig:1
# ## between-area
# between_min=0.0001 # orig:0.1
# between_max=0.07#0.225 # orig:1
# ## stimulation strength
# stim_strength= 175 #single area: 175
# ====> Shape looks good but a bit too high activation auditory testing (200) provides semantic but too high (50 + ) aftre 50 presentation it stays for auditory system only
# ==> increase the noise from 0.5 to 5 and increase global inhibition from 3.3 to 4 


## Problem comes from Global inhibition
# The global inhibition starts with 2 stp delay after the activation of excitatory neurons
# While in Felix it is only 1 stp
# During this 2 stp, the activation coulg go so high that will always provide merging what ever the parameters are


## New test ==> reduce maximum value of weight 0.2 instead of 0.225
## And increase Excitatory SJrec 40 ==> not link between central areas
## ==> Jrec = 42 ==> not enough
## ==> Jrec = 50 ==> too much
## ==> Jrec = 45 ==> not enough
##  ==> Jrec = 100 and Jslow = 46 ==> works but inhibition too high and influence the next presentation
## ON GOING ==> Jrec = 100 and Jslow = 10 ==>






##

#sJslow= 3.3#3.3# 3.3 #4.6 before #single_area: 24. # orig:55, modified: 55./4, good vals: 20-25
#sJinh= 60  #single area: 6
#Jexcitatory = 60
#k_2 = 0.5 # ==> Need to increase a bit the noise
#tauslow = 12
#
### within-area
#e_e_min=0.0001 # orig:0.1
#e_e_max= 0.07 #0.225 # orig:1
### between-area
#between_min=0.0001 # orig:0.1
#between_max=0.07#0.225 # orig:1
### stimulation strength
#stim_strength= 175 #single area: 175

## P1 ==> Looks good but no activation after 100 pres to more than the auditory system
## ==> Problem is no link between PB and PML. There are still some learning and will surely trigger some merging
## ==> Also present of burst between 2 presentations ===> need to reduce the time between two presentations from 150 to 50
## ==> increase a bit the noise from 0.5 to 1

## P2 ==> It works well, but one CA is taking over. Keep the same parameters, but increase the nb of patttern to 14 to increase competition and prevent merging

## P3 ==> With 14 patterns works really well, at 200 pres still no merging, begining of semantic category in primary extra areas
#==> activation remains low around 5 neurons in other regions than auditory
# ==> after 500 presentations only one big CA, the rest looks nice
# ==> maybe reduce a bit stimulation strength?
# ==> New funtion for pattern creation, it existed some overlap ! let's try with the same parameters but without overlap

## P4 ==> Some instability for some patterns with high level after 500 presentations, some pattern are too high
# ==> require more stability? Possible that the testing paradigme with only one stim is not enough, try with 2 stim
# ==> at P200 it works better but secondary areas in extra-sylvian region less strong than primary of extra
# Problem is that there is an explosion of activation after 200 presentation (probably 300 already)
# We should give less activation for each neurons, but increase the max size of connections


# P5 ==> max size stay at 0.2 but Jexcitatory and sJinh == 55
# ==> 200, 300 small activation in extra-sylvian region
# ==> weights for PB - PFI capped at 200 ==> as in Felix
# ==> small improvement after 400 presentations but max activity in extra-sylvian region capped at 5 neurons or less
# require a bit more activity ==> increase max of possible weight to 0.225


# P6 ==> max size increase to 0.225 but Jexcitatory and sJinh == 55
# 200 pres ==> works well but again small activation in extra-sylvian (5 max)
# 300 ==> works also well, limited to 8 in extra-sylvian but possible to have some merging at some point
# Try to reduce stim strength

#P7 stim_strength from 175 to stim_strength = 170
# Reduce to 12 patterns
# Mergining before 200 pattern, some pattern are triggering huge activity

#P8 Increase noise strength before stimulus from 0.1 to 5 and k2 from 1 to 3
# Works but after 200 pres some pattern start to be quite big and will probably lead to instability

#P9 increase noise k2 from 3 to 5 and decrease Jexcitatory and sJinh from 55 to 52
# ==> merging after 300 pres




if __name__ == "__main__":  
    tic = time.time()
    f = FelixNet()
    f.build_net()
    toc = time.time()
    total_training = 2010
    print("Noise: "+str(k_2))


    nest.overwrite_files = True   # easier for debugging

    print(f"Build Time           : {toc-tic:.1f} s")
    #w = np.array(f.areas['M1_L'].e_e_syn.get('weight'))
    #plt.figure(figsize=(12, 4))
    #plt.hist(w, bins=200);

    ## Change the number of pattern (it will split equally between object and action)
    nb_pattern = 12
    size_pattern = 19
    seed = 12
    motor, visu, audi, arti  = create_act_obj_pattern(nb_pattern, size_pattern, seed)
    
    tic = time.time()
    
    t_on = int(16*1)#0.5)#*2
    t_off = int(30*1)#0.5)#*2
    
    
    #nest.SetKernelStatus({'overwrite_files': False})
    f.train_action_object(motor, visu, audi, arti, num_reps=total_training, t_on=t_on, t_off=t_off, stim_specs=None, nb_pattern=nb_pattern)

    #nest.SetKernelStatus({'overwrite_files': True})

    toc = time.time()
    print(f"Train Time        : {toc-tic:.1f} s")
    
    print(f"Sim Time             : {time.time()-toc:.1f} s")

    print(f"Number of neurons    : {nest.network_size:_}")
    print(f"Number of connections: {nest.num_connections:_}")

    # Note on plot:
    #  - Data is written to one file per thread, globbing collects data from all files
    #  - Vertical axis in raster plot is global neuron ID
    #     - Each area occupies 1251 IDs, 625 excitatory, 625 inhibitory, 1 global inhibitory
    #     - For more meaningful plotting, one needs to write plot scripts that can translate
    #       the global IDs into more meaningful per-area indices or at least indicies
    #       without gaps.
    nest.raster_plot.from_file([str(p) for p in Path('.').glob('felix-*.dat')])
    
    w = np.array(f.areas['M1_L'].e_e_syn.get('weight'))
    plt.figure(figsize=(12, 4))
    plt.hist(w, bins=200);
    
    plt.show()
