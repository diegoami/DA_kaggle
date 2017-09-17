

import sys
sys.path.append('..')
from kaggle_blender import  merge_results
import pandas as pd
import numpy as np

#merge_results(['out/gbc_nes_1000_mad_t_8_1000.0.csv' , 'out/rfc_maf_10_nes_1000_1000.0.csv'], 'out/gbc_nes_1000_mad_t_8_1000_rfc_maf_10_nes_1000_1000.csv' )
merge_results(['out/grid_rfc_100.csv', 'out/xgbregressor_4_mad_5_mcw_3_gamma_0_cbt_09_ssa_07_a_1e3_lr_001_nes=5000100.csv'],
              'out/grid_rfc_100_xgbregressor_4_mad_5_mcw_3_gamma_0_cbt_09_ssa_07_a_1e3_lr_001_nes=5000100.csv' )
