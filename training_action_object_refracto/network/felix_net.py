import nest
import random
import numpy as np
import pandas as pd
import pickle
from .area import Area
from config import between_min, between_max
from utils.file_operations import ensure_directory_exists
from functions.function_annexe import stim_specs_patt_no, save_plot_weight, save_plot_activation_new, dat_from_file, sum_arrays

class FelixNet:
    def __init__(self):
        print("Initializing FelixNet")

        nest.set_verbosity("M_ERROR")

        try:
            nest.ResetKernel()
            nest.set(resolution=0.5, local_num_threads=16, rng_seed=12)
        except Exception as e:
            print("Error during NEST kernel reset:", e)

    def build_net(self):
        self.areas = {area: Area(area) for area in ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L',
                                                    'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']}

        ## 🔹 DEBUG: Print number of neurons in each area
        print("\n✅ Checking Neuron Counts in Each Area:")
        for area_name, area in self.areas.items():
            try:
                num_exc = len(area.exc)
                num_inh = len(area.inh)
                num_glob = len(area.glob)
                num_pg = len(area.pg) if hasattr(area, 'pg') else 0
                print(f"📌 Area {area_name}: Excitatory={num_exc}, Inhibitory={num_inh}, Global={num_glob}, PG={num_pg}")
            except AttributeError as e:
                print(f"❌ ERROR in area {area_name}: {e}")

        print("✅ All areas initialized\n")

        ## 🔹 Establish connections
        self.connect_areas()
        self.connect_recorders()

                                                
    def neurons2IDs(self, neurons):
        """
        Translates neuron numbers from 1 to 625 to their corresponding ID within an area of the model.

        neurons -- list of neuron numbers
        """
        return sorted(neurons)
        
    def stimulation_on(self, stim_specs):
        for area, specs in stim_specs.items():
            self.areas[area].stimulation_on(**specs)

    def stimulation_off(self):
        for area in self.areas.values():
            area.stimulation_off()
            
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


    def store(self, filename, motor, visu, audi, arti):
        """
        Store neuron membrane potential and synaptic weights to a file.
        """

        assert nest.NumProcesses() == 1, "Cannot dump MPI parallel"

        ## Store excitatory, inhibitory, and global inhibition values
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

        ## Store all synaptic weights
        weights = nest.GetConnections().get(
            ("delay", "receptor", "source", "synapse_model", "target", "weight"), output="pandas"
        )

        network = {
            "weights": weights,
            "pattern_motor": motor,
            "pattern_visual": visu,
            "pattern_auditory": audi,
            "pattern_articulatory": arti,
            "excitatory_neurons": excitatory_neurons,
            "inhibitory_neurons": inhibitory_neurons,
            "global_inhibition": global_inhibition,
        }

        directory = "./save_network/"
        print("SAVING NETWORK ==> ", filename)
        ensure_directory_exists(directory)

        with open(directory + filename, "wb") as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)

    def train_action_object(self, motor, visu, audi, arti, num_reps=10, t_on=16, t_off=30, stim_specs=None, nb_pattern=2, stim_strength=500):
    
        ensure_directory_exists("./plot_weight")
        ensure_directory_exists("./save_network")


        nest.SetKernelStatus({'overwrite_files': True})
        patt_no = 0
        self.store("network_0", motor, visu, audi, arti)
        gi_tot = []
        patt_no_count = [0] * nb_pattern
        count_firs_pres = 0

        while any(count < num_reps for count in patt_no_count):
            with nest.RunManager():
                patt_no = random.randint(0, nb_pattern - 1)
                if patt_no_count[patt_no] >= num_reps:
                    continue

                print(f"################\nPresentation patt_no: {patt_no}")
                nest.SetKernelStatus({'overwrite_files': True})

                stim_specs = stim_specs_patt_no(self, patt_no, nb_pattern, motor, visu, audi, arti, stim_strength)
                patt_no_count[patt_no] += 1

                

                self.stimulation_on(stim_specs)
                for _ in range(t_on):
                    nest.Run(0.5)
                self.stimulation_off()


                ## Wait until the GI goes below a given threshold
                counter_stim_pause = 0
                self.stimulation_off()
                gi_PB = self.areas["PB"].glob.get(output="pandas")["V_m"].values[0]
                gi_PF_i = self.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]
                while ((gi_PB > 0.70) | (gi_PF_i > 0.70) | (counter_stim_pause < t_off)):
                        nest.Run(0.5)
                        gi_PB = self.areas["PB"].glob.get(output="pandas")["V_m"].values[0]
                        gi_PF_i = self.areas["PF_i"].glob.get(output="pandas")["V_m"].values[0]
                        counter_stim_pause += 0.5


                # Save progress every 30 presentations
                if np.sum(patt_no_count) % 5 == 0: ## NEED TO CHANGE FIRST 0 TO 30
                    dat = dat_from_file('felix-*.dat')
                    dat['sum'] = dat['matrix'].apply(sum_arrays)
                    dat["Pres"] = patt_no_count[patt_no]
                    dat["patt_no"] = patt_no
                    save_plot_activation_new(patt_no_count[patt_no], dat, patt_no)

                # Save network at specific intervals
                if (patt_no_count[-1] in [10, 30, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1700, 2000]) and (patt_no == nb_pattern - 1):
                    self.store(f"network_{patt_no_count[-1]}", motor, visu, audi, arti)

        save_plot_weight(self, patt_no_count[-1])
        dat['sum'] = dat['matrix'].apply(sum_arrays)
        save_plot_activation_new(patt_no_count[-1], dat, patt_no)
