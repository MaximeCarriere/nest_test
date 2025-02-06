import nest
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm  # For a nice progress bar
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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
        Create inter-area connections and print a structured connectivity summary with actual connection counts.
        """

        # Store connection counts
        connection_counts = defaultdict(int)

        connectome = [
            # Visual System
            ('V1', 'TO'), ('V1', 'AT'), ('TO', 'AT'),
            # Motor System
            ('PF_L', 'PM_L'), ('PF_L', 'M1_L'), ('PM_L', 'M1_L'),
            # Auditory System
            ('A1', 'AB'), ('A1', 'PB'), ('AB', 'PB'),
            # Articulatory System
            ('PF_i', 'PM_i'), ('PF_i', 'M1_i'), ('PM_i', 'M1_i'),
            # Cross system next-neighbour
            ('AT', 'PF_L'), ('AT', 'PB'), ('AT', 'PF_i'),
            ('PF_L', 'PB'), ('PF_L', 'PF_i'), ('PB', 'PF_i'),
            # Cross system jumping links
            ('TO', 'PF_L'), ('AT', 'PM_L'), ('AB', 'PF_i'), ('PB', 'PM_i')
        ]

        cross_system = [
            ('AT', 'PF_L'), ('AT', 'PB'), ('AT', 'PF_i'),
            ('PF_L', 'PB'), ('PF_L', 'PF_i'), ('PB', 'PF_i'),
            ('TO', 'PF_L'), ('AT', 'PM_L'), ('AB', 'PF_i'), ('PB', 'PM_i')
        ]

        within_system = [
            ('V1', 'TO'), ('V1', 'AT'), ('TO', 'AT'),
            ('PF_L', 'PM_L'), ('PF_L', 'M1_L'), ('PM_L', 'M1_L'),
            ('A1', 'AB'), ('A1', 'PB'), ('AB', 'PB'),
            ('PF_i', 'PM_i'), ('PF_i', 'M1_i'), ('PM_i', 'M1_i')
        ]

        # If you do not want reciprocal connections, comment the next two lines
        cross_system.extend([(tgt, src) for src, tgt in cross_system])
        within_system.extend([(tgt, src) for src, tgt in within_system])

        kc_cross = 0.13 * 0.5
        kc_within = 0.13

        print("\n🔗 **Creating Inter-Area Connections...**")

        # **Cross-System Connections**
        print("\n🌐 **Cross-System Connections:**")
        for src, tgt in tqdm(cross_system, desc="Connecting Cross-System Areas"):
            nest.Connect(self.areas[src].exc, self.areas[tgt].exc,
                         {'rule': 'pairwise_bernoulli',
                          'p': kc_cross * nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                                                                                nest.spatial.distance.y,
                                                                                std_x=9, std_y=9, mean_x=0,
                                                                                mean_y=0, rho=0),
                          'mask': {'grid': {'shape': [19, 19]}, 'anchor': [9, 9]}},
                         {'synapse_model': 'abs_synapse', 'receptor_type': 1,
                          'weight': nest.random.uniform(between_min, between_max), 'delay': 1})

        # **Within-System Connections**
        print("\n🏠 **Within-System Connections:**")
        for src, tgt in tqdm(within_system, desc="Connecting Within-System Areas"):
            nest.Connect(self.areas[src].exc, self.areas[tgt].exc,
                         {'rule': 'pairwise_bernoulli',
                          'p': kc_within * nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                                                                                 nest.spatial.distance.y,
                                                                                 std_x=9, std_y=9, mean_x=0,
                                                                                 mean_y=0, rho=0),
                          'mask': {'grid': {'shape': [19, 19]}, 'anchor': [9, 9]}},
                         {'synapse_model': 'abs_synapse', 'receptor_type': 1,
                          'weight': nest.random.uniform(between_min, between_max), 'delay': 1})

        # **Retrieve and Count Connections**
        print("\n📊 **Retrieving Connection Data...**")
        network_weights = nest.GetConnections().get(
            ("source", "target", "weight"), output="pandas"
        )

        
        # Compute actual number of connections per area pair
        for (src, tgt) in cross_system + within_system:
            try:
                # ✅ Dynamically fetch excitatory neurons for each area
                n_area1 = self.areas[src].exc.get(output="pandas")
                n_area2 = self.areas[tgt].exc.get(output="pandas")

                # ✅ Filter weight data to count connections from src to tgt
                weight_data = network_weights[
                    (network_weights["source"].isin(n_area1.index)) &
                    (network_weights["target"].isin(n_area2.index))
                ]

                connection_counts[(src, tgt)] = len(weight_data)

            except KeyError as e:
                print(f"⚠️ Warning: Could not fetch data for {src} → {tgt}: {e}")

        # **Summary of Connections**
        print("\n📋 **Summary of Connections**")
        print("══════════════════════════════════════════════════")
        print(f"{'Source Area':<10} → {'Target Area':<10} | {'# Connections':<10}")
        print("══════════════════════════════════════════════════")

        heatmap_area = []
        for (src, tgt), count in connection_counts.items():
            print(f"{src:<10} → {tgt:<10} | {count:<10}")
            heatmap_area.append([src, tgt, count])
            
        heatmap_area = pd.DataFrame(heatmap_area, columns=["Area1","Area2", "NB_Connections"])
        heatmap_area = heatmap_area.pivot(index="Area1",columns="Area2",values="NB_Connections")
        
        print("══════════════════════════════════════════════════")
        print("✅ **All Connections Established Successfully!**\n")
        
        # Define the specific order of areas
        area_order = ['V1', 'TO', 'AT', 'PF_L', 'PM_L', 'M1_L', 'A1', 'AB', 'PB', 'PF_i', 'PM_i', 'M1_i']

        # Reorder rows and columns based on the specific order
        heatmap_area = heatmap_area.reindex(index=area_order, columns=area_order)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_area, annot=True, fmt=".0f", cmap="viridis", linewidths=0.5)
        plt.title("Inter-Area Connection Heatmap")
        plt.xlabel("Target Area")
        plt.ylabel("Source Area")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.savefig('./plot_training/heat_map_area.png')
        print("══════════════════════════════════════════════════")
        print("✅ **Heatmap Area saved in ./plot_training/heat_map_area.png!**\n")
        


        

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
        Store neuron membrane potential and synaptic weights to given file with a progress bar.
        """
        print("\n💾 SAVING NETWORK ==> ", filename)
        assert nest.NumProcesses() == 1, "Cannot dump MPI parallel"

        ## 🟢 Inhibitory Parameters
        print("\n🔹 Extracting Inhibitory Parameters...")
        inh_neurons_param_list = ["k_1", "tau_m"]
        list_param_value_inh = []
        for param in tqdm(inh_neurons_param_list, desc="Processing Inhibitory Parameters"):
            value = list(set(self.areas['A1'].inh.get([param], output="pandas")[param].values.tolist()))
            list_param_value_inh.append([param, value])
        list_param_value_inh = pd.DataFrame(list_param_value_inh, columns=["param", "value"])

        ## 🟢 Excitatory Parameters
        print("\n🔹 Extracting Excitatory Parameters...")
        exc_neurons_param_list = ["om", "alpha", "alpha_e", "tau_adapt", "k_2", "Jexcitatory", "tau_m"]
        list_param_value = []
        for param in tqdm(exc_neurons_param_list, desc="Processing Excitatory Parameters"):
            value = list(set(self.areas['A1'].exc.get([param], output="pandas")[param].values.tolist()))
            list_param_value.append([param, value])
        list_param_value = pd.DataFrame(list_param_value, columns=["param", "value"])

        ## 🟢 Global Inhibition Parameters
        print("\n🔹 Extracting Global Inhibition Parameters...")
        glob_neurons_param_list = ["k_1", "tau_m"]
        list_param_value_glob = []
        for param in tqdm(glob_neurons_param_list, desc="Processing Global Inhibition Parameters"):
            value = list(set(self.areas['A1'].glob.get([param], output="pandas")[param].values.tolist()))
            list_param_value_glob.append([param, value])
        list_param_value_glob = pd.DataFrame(list_param_value_glob, columns=["param", "value"])

        ## 🟢 Store Neurons Data
        print("\n🔹 Extracting Neuron Data...")

        excitatory_neurons = []
        inhibitory_neurons = []
        global_inhibition = []

        for area in tqdm(self.areas.keys(), desc="Processing Areas"):
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

        ## 🟢 Store Synaptic Weights
        print("\n🔹 Extracting Synaptic Weights...")
        test = nest.GetConnections().get(
            ("delay", "receptor", "source", "synapse_model", "target", "weight"), output="pandas"
        )

        ## 🟢 Save Network Data
        network = {
            "param_excitatory": list_param_value,
            "param_inhibitory": list_param_value_inh,
            "param_global": list_param_value_glob,
            "weight": test,
            "pattern_motor": motor,
            "pattern_visual": visu,
            "pattern_auditory": audi,
            "pattern_articulatory": arti,
            "excitatory_neurons": excitatory_neurons,
            "inhibitory_neurons": inhibitory_neurons,
            "global_inhibition": global_inhibition,
        }

        directory = "./save_network/"
        ensure_directory_exists(directory)

        with open(directory + filename, "wb") as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)

        print("✅ Saving Complete!")



    def train_action_object(self, motor, visu, audi, arti, num_reps=10, t_on=16, t_off=30, stim_specs=None, nb_pattern=2, stim_strength=500):
    
        ensure_directory_exists("./save_network")
        ensure_directory_exists("./processing_data")


        nest.SetKernelStatus({'overwrite_files': True, 'data_path': "./processing_data"})
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
                    dat = dat_from_file('./processing_data/felix-*.dat')
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
