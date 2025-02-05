
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
        cross_system.extend(list((tgt, src) for src, tgt in cross_system))

        kc_cross = 0.13*0.5

        # Now create connections
        for src, tgt in cross_system:
            nest.Connect(self.areas[src].exc, self.areas[tgt].exc,
                        {'rule': 'pairwise_bernoulli',
                        'p': kc_cross* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
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

        kc_within = 0.13

        # Now create connections
        for src, tgt in within_system:
            nest.Connect(self.areas[src].exc, self.areas[tgt].exc,
                        {'rule': 'pairwise_bernoulli',
                        'p':   kc_within* nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
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


        for area in self.areas.values():
            nest.Connect(area.exc, self.spike_rec)
            nest.Connect(self.vm, area.exc)
            
    def build_net(self):
        """
        Build network
        1. All areas with internal connections
        2. Connections between areas
        3. Connect recorders
        """
        

                                                
        self.areas = {area: Area(area) for area in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L',
                                                'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']}
                                                
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

                    gi_AB = f.areas["AB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                    gi_PM_i = f.areas["PM_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                    

                    self.stimulation_on(stim_specs)
                    counter_run = 0
                    for time_to_run in range(0, t_on):
                        counter_run = counter_run + 0.5
                        nest.Run(0.5)

                        
                        

                    counter_stim_pause = 0
                    self.stimulation_off()

                
                    gi_PB = f.areas["PB"].glob.get(output="pandas")["V_m"].values[0]#*sJslow
                    gi_PF_i = f.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]#*sJslow


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
                            gi_tot.append([counter_stim_pause_init+count_firs_pres+counter_run+counter_stim_pause,area_eph,gi_eph, "stim_off",999999, np.sum(patt_no_count)])



                    count_firs_pres = count_firs_pres+counter_run+counter_stim_pause


                

                if np.sum(patt_no_count)%30==0:
                     dat=dat_from_file('felix-*.dat')
                     dat['sum'] = dat['matrix'].apply(sum_arrays)
                     dat["Pres"] = patt_no_count[patt_no]
                     dat["patt_no"] = patt_no

                     save_plot_activation_new(patt_no_count[patt_no], dat, patt_no)
                if (patt_no_count[-1] in [1, 10, 30, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1700, 2000])&(patt_no==nb_pattern-1):
                    f.store("network_"+str(patt_no_count[-1]), motor, visu, audi, arti)
                    

        dat['sum'] = dat['matrix'].apply(sum_arrays)
        dat["Pres"] = num_reps
        save_plot_weight(f, patt_no_count[-1])
        save_plot_activation_new(patt_no_count[-1], dat, patt_no)
                    


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






if __name__ == "__main__":  
    tic = time.time()
    f = FelixNet()
    f.build_net()
    toc = time.time()
    total_training = 2010
    print("Noise: "+str(k_2))


    nest.overwrite_files = True   # easier for debugging

    print(f"Build Time           : {toc-tic:.1f} s")

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

