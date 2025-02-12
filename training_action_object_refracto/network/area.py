import nest
from config import *


# Ensure felixmodule is installed before creating neurons
if "felix_exc" not in nest.Models():
    print("Felix Module not found. Installing now...")
    nest.Install('felixmodule')

class Area:
    def __init__(self, name):
        self.name = name



        pos = nest.spatial.grid(shape=[EXC_NEURONS, EXC_NEURONS], extent=[EXC_NEURONS, EXC_NEURONS], edge_wrap=True)
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


        # Inh -> Exc connections
        nest.Connect(self.inh, self.exc,
                 'one_to_one', syn_spec={'synapse_model': 'static_synapse', 'receptor_type': 2, 'delay': 0.5,
                 'weight':-1*sJinh})
        # global inhibition
        nest.Connect(self.exc, self.glob, syn_spec={'delay':0.5, 'weight': 1})
        nest.Connect(self.glob, self.exc, syn_spec={'delay':0.5, 'receptor_type': 2, 'weight':-1*sJslow})

        self.exc_stim = None


    def stimulation_on(self, neurons, I_stim):
        self.exc_stim = self.exc[neurons]
        self.exc_stim.I_e = I_stim

    def stimulation_off(self):
        if self.exc_stim:
            self.exc_stim.I_e = 0
            self.exc_stim = None
