import nest
import pickle
import pandas as pd
import numpy as np  # Ensure numpy is available
from utils.utils import ensure_directory_exists
from config import TESTING_OUTPUT_DIR


class Restore_Area:
    """ Recreates a neural area from a saved network file. """

    def __init__(self, directory_network, area):
        try:
            with open(directory_network, "rb") as f:
                network = pickle.load(f)  # Load normally

        except ModuleNotFoundError as e:
            if "numpy" in str(e):  # NumPy module issue
                print("⚠️ Warning: NumPy is missing, trying to load without it.")
                with open(directory_network, "rb") as f:
                    network = pickle.load(f, encoding="latin1")  # Alternative load
            
                # Convert NumPy arrays to Python lists
                for key in network.keys():
                    if isinstance(network[key], np.ndarray):
                        network[key] = network[key].tolist()

            else:
                raise e  # Reraise other errors

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
        """ Apply stimulation to specific neurons. """
        self.exc_stim = self.exc[neurons]
        self.exc_stim.I_e = I_stim

    def stimulation_off(self):
        """ Turn off stimulation. """
        if self.exc_stim:
            self.exc_stim.I_e = 0
            self.exc_stim = None
