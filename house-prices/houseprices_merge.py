

import sys
sys.path.append('..')
from kaggle_blender import  merge_results
import pandas as pd
import numpy as np

merge_results(['out/gbc_nes_1000_mad_t_8_1000.0.csv' , 'out/rfc_maf_10_nes_1000_1000.0.csv'], 'out/gbc_nes_1000_mad_t_8_1000_rfc_maf_10_nes_1000_1000.csv' )
