# config.py

# Define parameters for testing
NUM_REPS = 10       # Number of repetitions for testing
T_ON = 2           # Time stimulus is ON
T_OFF = 30          # Time stimulus is OFF
STIM_STRENGTH = 500 # Strength of stimulation
TESTING_OUTPUT_DIR = "./testing_data"

# List of trained networks to be tested
NETWORKS = [
    "save_network"
]

# Directory where networks are stored
NETWORKS_DIR = "/Users/maximecarriere/src/nest_test/training_action_object_refracto/"

# ✅ Define the list of training steps for each network
NETWORKS_LIST = [10, 1000]
