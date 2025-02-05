import time
import nest

# ✅ Only install if it's not already loaded
installed_modules = nest.GetKernelStatus().get("loaded_modules", [])
if "felixmodule" not in installed_modules:
    nest.Install('felixmodule')
else:
    print("Felix Module is already installed, skipping installation.")

from network.felix_net import FelixNet  # Import AFTER loading felixmodule
from utils.file_operations import create_act_obj_pattern, ensure_directory_exists, show_owerlapp_pattern
from config import TOTAL_TRAINING, NB_PATTERN, SIZE_PATTERN, SEED, stim_strength


if __name__ == "__main__":
    tic = time.time()
    f = FelixNet()
    f.build_net()
    toc = time.time()
    
    ensure_directory_exists("./plot_training")
    print(f"Build Time: {toc-tic:.1f} s")

    motor, visu, audi, arti = create_act_obj_pattern(NB_PATTERN, SIZE_PATTERN, SEED)
    
    tic = time.time()
    f.train_action_object(motor, visu, audi, arti, num_reps=TOTAL_TRAINING, stim_strength=stim_strength, nb_pattern=NB_PATTERN)


    toc = time.time()
    print(f"Train Time: {toc-tic:.1f} s")
