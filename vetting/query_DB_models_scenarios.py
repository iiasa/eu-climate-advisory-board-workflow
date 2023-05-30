# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:00:01 2023

@author: byers
"""

# Query all scenarios in database

import ixmp
import pandas as pd
import pyam
from vetting_functions import *


#%%

wd = 'C:\\Users\\byers\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\'


instance_s = 'eu_climate_submission'
config_s = f'{wd}euab_submission.properties'

instance_i = 'eu_climate_internal'
config_i = f'{wd}euab_internal.properties'


# Submissions

mp_s = ixmp.Platform(dbprops=config_s)
sl_s = mp_s.scenario_list()

models_s = sl_s.model.unique()

sl_s['model_stripped'] = sl_s.reset_index()['model'].apply(strip_version).values

#%%
# Internal
mp_i = ixmp.Platform(dbprops=config_i)
sl_i = mp_i.scenario_list()

models_i = sl_i.model.unique()

sl_i['model_stripped'] = sl_i.reset_index()['model'].apply(strip_version).values


#%%
# Models in submission not in internal
mod_sub = list(set(models_s) - set(models_i))
# ['REMIND 3.1', 'RECC 2.4', 'TIMES-Ireland Model v1.0', 'OSeMBE v1.0.0']
# REMIND 3.1 is duplicate of 3.2, so ignore. # 3 models submitted but not further considered as lacking regional information / country level only.
# TIMES-Ireland - country only
# RECC 2.4 - only 3 variables
# OSeMBE v1.0.0 - no emissions variables
instance = 'eu_climate_submission'
instance = 'eu-climate-advisory-board-internal'

# qp = pyam.read_iiasa(instance,
#                         model='OSeMBE v1.0.0',
#                         meta=False)

# Models in internal not in submission
mod_int = list(set(models_i) - set(models_s))
# 'WITCH 5.1', 'TIAM-ECN 1.2', 'Euro-Calliope 2.0', 'REMIND 3.2'
# Additional scenarios brought in for ECEMF
mylist = list(dict.fromkeys(list(models_i)+list(models_s)))

# Total of 54 models
# N.B. REMIND 3.1 is the first version of the extra REMIND 3.2 scenarios
# 52 models submitted, + REMIND 3.2 solicited =  53
minus 3 above




