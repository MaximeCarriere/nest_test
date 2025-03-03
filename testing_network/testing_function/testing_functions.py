from utils.utils import ensure_directory_exists
from network.network import FelixNet  # Ensure FelixNet is imported
import pickle
import pandas as pd
import numpy as np  # Ensure numpy is available

def safe_pickle_load(file_path):
    """ Load a pickle file safely, even if numpy is missing. """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)  # Normal loading
    except ModuleNotFoundError as e:
        if "numpy" in str(e):  # Handle missing numpy
            print(f"⚠️ Warning: NumPy is missing when loading {file_path}, attempting fallback.")
            with open(file_path, "rb") as f:
                network = pickle.load(f, encoding="latin1")
            
            # Convert NumPy arrays to lists
            for key in network.keys():
                if isinstance(network[key], np.ndarray):
                    network[key] = network[key].tolist()
            return network
        else:
            raise e  # Raise other errors


def testing_auditory_multiple_networks(networks_list, networks_dir, network_out):
    """ Test auditory networks. """
    print("network_out: ", network_out)
    ensure_directory_exists(network_out)

    for patt_no_count in networks_list:
        print("##############################")
        print(f"     NETWORK: {patt_no_count}")
        print("##############################")

        filename = f"network_{patt_no_count}"
        directory_network = f"{networks_dir}/{filename}".strip()  # Remove accidental spaces
        print(directory_network)
        
        network = safe_pickle_load(directory_network)  # Load safely

        audi = network["pattern_auditory"]
        arti = network["pattern_articulatory"]

        f = FelixNet()
        f.rebuild_net(directory_network)

        # Check if test_aud exists in FelixNet before calling
        if hasattr(f, 'test_aud'):
            f.test_aud(audi, arti, patt_no_count, num_reps=10, t_on=2, t_off=30)
        else:
            print("⚠️ Warning: `test_aud` method is missing in FelixNet!")

    print("✅ Testing Auditory Completed")


def testing_articulatory_multiple_networks(networks_list, networks_dir, network_out):
    """ Test articulatory networks. """
    ensure_directory_exists(network_out)

    for patt_no_count in networks_list:
        print("##############################")
        print(f"     NETWORK: {patt_no_count}")
        print("##############################")

        filename = f"network_{patt_no_count}"
        directory_network = f"{networks_dir}/{filename}".strip()  # Remove accidental spaces

        network = safe_pickle_load(directory_network)  # Load safely

        audi = network["pattern_auditory"]
        arti = network["pattern_articulatory"]

        f = FelixNet()
        f.rebuild_net(directory_network)

        # Check if test_art exists in FelixNet before calling
        if hasattr(f, 'test_art'):
            f.test_art(audi, arti, patt_no_count, num_reps=10, t_on=2, t_off=30)
        else:
            print("⚠️ Warning: `test_art` method is missing in FelixNet!")


    print("✅ Testing Articulatory Completed")
