# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:42:59 2023

@author: byers
"""
# process_region_mapping.py

# process mapping files
import ixmp
import pandas as pd

folder_reg_excel = f''




#%% See what models on server
instance = 'eu_climate_submission'
config = 'C:\\Users\\byers\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\euab.properties'

mp = ixmp.Platform(dbprops=config)
sl = mp.scenario_list()

models = sl.model.unique()


#%% process excel registration files
isolist = ['BRA','CHN','....]
df = pd.DataFrame(index=isolist)
files = glob.glob('?.xlsx')
for f in files:
    
     
    
    
    dfin = pd.read_excel(f, sheet_name='...')[['ISO','native']]
    dfin.set_index('ISO', inplace=True)
    df.join(dfin, on='index')

