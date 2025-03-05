# stim.py
import random
from config import STIM_STRENGTH

#def stim_specs_patt_no(patt_no, nb_pattern, motor, visu, audi, arti):
#    """ Define stimulation specifications based on pattern number """
#    if (patt_no + 1 <= (nb_pattern / 2)):
#        return {
#            'V1': {'neurons': visu[patt_no], 'I_stim': STIM_STRENGTH},
#            'M1_L': {'neurons': random.sample(list(range(0, 625)), 19), 'I_stim': STIM_STRENGTH},
#            'A1': {'neurons': audi[patt_no], 'I_stim': STIM_STRENGTH},
#            'M1_i': {'neurons': arti[patt_no], 'I_stim': STIM_STRENGTH}
#        }
#    else:
#        return {
#            'V1': {'neurons': random.sample(list(range(0, 625)), 19), 'I_stim': STIM_STRENGTH},
#            'M1_L': {'neurons': motor[patt_no], 'I_stim': STIM_STRENGTH},
#            'A1': {'neurons': audi[patt_no], 'I_stim': STIM_STRENGTH},
#            'M1_i': {'neurons': arti[patt_no], 'I_stim': STIM_STRENGTH}
#        }



def stim_specs_patt_no_gui(auditory_input,
                articulatory_input,
                visual_input,
                motor_input,
                patt_no,
                num_reps,
                t_on,
                t_off,
                auditory_check,
                articulatory_check,
                visual_check,
                motor_check,
                stim_strength):
                
    
    stim_specs={}
    if auditory_check==True:
            stim_specs['A1'] =  {'neurons': auditory_input[patt_no],
                            'I_stim':   stim_strength}
                            
    if articulatory_check==True:
            stim_specs['M1_i'] =  {'neurons': articulatory_input[patt_no],
                            'I_stim':   stim_strength}
    
    if motor_check==True:
            stim_specs['M1_L'] =  {'neurons': motor_input[patt_no],
                            'I_stim':   stim_strength}
                            
    if visual_check==True:
            stim_specs['V1'] =  {'neurons': visual_input[patt_no],
                            'I_stim':   stim_strength}
                            
                            
    return stim_specs
        





