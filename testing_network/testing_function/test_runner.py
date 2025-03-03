import os
from config import NETWORKS, NETWORKS_DIR, TESTING_OUTPUT_DIR, NETWORKS_LIST, TEST_MODE
from utils.utils import ensure_directory_exists
from testing_function.testing_functions import (
    testing_auditory_multiple_networks,
    testing_articulatory_multiple_networks,
)


def run_all_tests():
    """Run tests on all saved networks based on TEST_MODE in config.py."""
    print("🚀 Starting FelixNet Testing...")

    for network in NETWORKS:
        print(f"🧪 Testing Network: {network}")

        network_path = os.path.join(NETWORKS_DIR, network)
        output_path = os.path.join(TESTING_OUTPUT_DIR, f"testing_{network}")

        # Ensure output directory exists
        ensure_directory_exists(output_path)

        if TEST_MODE == "auditory":
            print(f"🎵 Running auditory tests for {network}")
            testing_auditory_multiple_networks(NETWORKS_LIST, network_path, output_path)

        elif TEST_MODE == "articulatory":
            print(f"🗣️ Running articulatory tests for {network}")
            testing_articulatory_multiple_networks(NETWORKS_LIST, network_path, output_path)

        elif TEST_MODE == "both":
            print(f"🎵🗣️ Running both auditory and articulatory tests for {network}")
            testing_auditory_multiple_networks(NETWORKS_LIST, network_path, output_path)
            testing_articulatory_multiple_networks(NETWORKS_LIST, network_path, output_path)

    print("✅ All tests completed!")
