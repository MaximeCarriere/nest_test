## Verbose (print)
verbose = 0 #if verbose = 0 limited print || if 1 some print along the building/training

## Saving round
when_to_save = [10, 20, 50, 100] #When to save the network (number of presentation)

# Global Parameters
TOTAL_TRAINING = 100
NB_PATTERN = 12
SIZE_PATTERN = 19
SEED = 12

# Network Parameters
sJslow = 65
sJinh = 500
Jexcitatory = 500
k_2 = 50
e_e_min = 0.01
e_e_max = 0.1
between_min = 0.01
between_max = 0.1
stim_strength = 500

# Excitatory Variables
k_1_exc = 0.01
tau_m_exc = 5
alpha_exc = 0.01
tau_adapt_exc = 20

# Inhibitory Variables
k1_inh = 1
tau_m_inh = 10

# Global inhibition
k1_glob = 1
tau_m_glob = 24


# Define area size dynamically
EXC_NEURONS = 25  # Change this value to set the size dynamically (e.g., 30x30)
INH_NEURONS = EXC_NEURONS  # Keep inhibitory neurons equal to excitatory
