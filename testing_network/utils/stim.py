# stim.py
import random
from config import STIM_STRENGTH

def stim_specs_patt_no(patt_no, nb_pattern, motor, visu, audi, arti):
    """ Define stimulation specifications based on pattern number """
    if (patt_no + 1 <= (nb_pattern / 2)):
        return {
            'V1': {'neurons': visu[patt_no], 'I_stim': STIM_STRENGTH},
            'M1_L': {'neurons': random.sample(list(range(0, 625)), 19), 'I_stim': STIM_STRENGTH},
            'A1': {'neurons': audi[patt_no], 'I_stim': STIM_STRENGTH},
            'M1_i': {'neurons': arti[patt_no], 'I_stim': STIM_STRENGTH}
        }
    else:
        return {
            'V1': {'neurons': random.sample(list(range(0, 625)), 19), 'I_stim': STIM_STRENGTH},
            'M1_L': {'neurons': motor[patt_no], 'I_stim': STIM_STRENGTH},
            'A1': {'neurons': audi[patt_no], 'I_stim': STIM_STRENGTH},
            'M1_i': {'neurons': arti[patt_no], 'I_stim': STIM_STRENGTH}
        }
