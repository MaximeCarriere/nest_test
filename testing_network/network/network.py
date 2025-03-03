# network.py
import nest
import pickle
import pandas as pd
from functions.function_annexe import *
from utils.utils import ensure_directory_exists
from network.restore_area import Restore_Area


class FelixNet:
    """ FelixNet Class for reconstructing and testing trained networks. """
    
    def __init__(self):
        print("Initializing FelixNet")

        nest.set_verbosity("M_ERROR")
        nest.ResetKernel()
        nest.set(resolution=0.5, local_num_threads=16, rng_seed=12345)
        ensure_directory_exists("./processing_data", clear=True)

    def rebuild_net(self, directory):
        """ Load and reconstruct the network from a saved file. """
        with open(directory, "rb") as f:
            network = pickle.load(f)

        print(network["excitatory_neurons"].area.unique())
        self.areas = {area: Restore_Area(directory, area) for area in
                    network["excitatory_neurons"].area.unique()}
        self.reconnect_areas(directory)
        self.connect_recorders()
        
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

            nest.Connect(weight_area_exc["source"].values, weight_area_exc["target"].values,
                             "one_to_one",
                             {"synapse_model":connection_type,
                              "weight": weight_area_exc["weight"].values,
                             "delay":weight_area_exc["delay"].values,
                              'receptor_type': 1
                             })#,
    

   


    def connect_recorders(self):
        """ Connect spike recorder and multimeter for analysis. """
        self.spike_rec = nest.Create('felix_spike_recorder', {'record_to': 'ascii', 'label': 'felix'})
        self.vm = nest.Create('multimeter', {'record_from': ["V_m"], 'record_to': 'ascii', 'label': 'V_m'})
        
        for area in self.areas.values():
            nest.Connect(area.exc, self.spike_rec)
            nest.Connect(self.vm, area.exc)
    def stimulation_on(self, stim_specs):
        for area, specs in stim_specs.items():
            self.areas[area].stimulation_on(**specs)

    def stimulation_off(self):
        for area in self.areas.values():
            area.stimulation_off()
            
            
    def test_aud(self, audi, arti,  patt_no_count, num_reps=10, t_on=16, t_off=30):
        ensure_directory_exists("./testing_data")
        
        stim_strength = 500
        nest.SetKernelStatus({'overwrite_files': True, 'data_path': "./processing_data"})

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
                            dat = dat_from_file('./processing_data/felix-*.dat')
                            dat['sum'] = dat['matrix'].apply(sum_arrays)
                            dat["stim"] = stim
                            dat["patt_no"]=patt_no
                            dat["time"] = dat["time"] - dat["time"].min()
                            dat["Presentation"] = patt_no_count
                            dataS.append(dat)
                   
        dataS =pd.concat(dataS)
        dataS["Cond"]="Audi"
        dataS.to_csv("./testing_data/audi_"+str(patt_no_count)+"_presentations.csv")
        
        
        
    def test_art(self, audi, arti,  patt_no_count, num_reps=10, t_on=16, t_off=30):
        ensure_directory_exists("./testing_data")
        stim_strength = 500
        nest.SetKernelStatus({'overwrite_files': True, 'data_path': "./processing_data"})

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
         
                            counter_stim_pause += 0.5

                        # Print values after update
                        print("gi_AB:", gi_AB)
                        print("gi_PM_i:", gi_PM_i)
                        print("counter_stim_pause:", counter_stim_pause)
                        print("#######################")
            
            
                        if iterations==1:
                            dat = dat_from_file('./processing_data/felix-*.dat')
                            dat['sum'] = dat['matrix'].apply(sum_arrays)
                            dat["stim"] = stim
                            dat["patt_no"]=patt_no
                            dat["time"] = dat["time"] - dat["time"].min()
                            dat["Presentation"] = patt_no_count
                            dataS.append(dat)
                    
        dataS =pd.concat(dataS)
        dataS["Cond"]="Arti"
        dataS.to_csv("./testing_data/arti_"+str(patt_no_count)+"_presentations.csv")


