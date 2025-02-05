import pickle
import os
import random

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_network(filename, network_data):
    ensure_directory_exists("./save_network/")
    with open(f"./save_network/{filename}", "wb") as f:
        pickle.dump(network_data, f, pickle.HIGHEST_PROTOCOL)



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

    for i in range(nb_pattern):
        motor.append(sorted(random.sample(list(neuron_pool_motor), size_pattern)))
        visu.append(sorted(random.sample(list(neuron_pool_visu), size_pattern)))
        audi.append(sorted(random.sample(list(neuron_pool_audi), size_pattern)))
        arti.append(sorted(random.sample(list(neuron_pool_arti), size_pattern)))

        neuron_pool_motor -= set(motor[-1])
        neuron_pool_visu -= set(visu[-1])
        neuron_pool_audi -= set(audi[-1])
        neuron_pool_arti -= set(arti[-1])

    return motor, visu, audi, arti

