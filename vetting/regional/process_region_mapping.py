# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:42:59 2023

@author: byers
"""
# process_region_mapping.py

# process mapping files
import ixmp
import pandas as pd
import yaml
import glob


wd = 'C:\\Users\\byers\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\'
folder_registrations = f'{wd}\\vetting\\regional\\input_data\\model_registrations\\'
folder_native_regions = f'{wd}\\vetting\\regional\\input_data\\model_native_regions\\'

#%% See what models on server
instance = 'eu_climate_submission'
config = f'{wd}euab.properties'

mp = ixmp.Platform(dbprops=config)
sl = mp.scenario_list()

models = sl.model.unique()


#%% Loop through yamls and add ISO-3

yamls = glob.glob(folder_native_regions+'POL*.y*ml')
for y in yamls:
    fn = y.split('\\')[-1]
    # model = fn.split('.yaml')[0]
    
    with open(y, "r") as stream:
        yin = yaml.safe_load(stream)
        
        idx = 0
        notes = None
        model = list(yin[0].keys())[0]
    
        try:
            # Open corresponding excel file
            print(f'trying {model}')

            modelstr = model.replace('/','-') #parse.quote_plus(model)
            reg_regions = pd.read_excel(f'{folder_registrations}EUAB_model_registration_{modelstr}.xlsx', sheet_name='regional definition')[['Native Region Name','ISO Code',]]
            reg_regions.columns = ['native_name','ISO',]
            reg_dic = reg_regions.groupby('native_name')['ISO'].apply(list).to_dict()
            print(f'opened {model}')

            for item in yin[0][model]:
                
                if type(item) is dict:
                    modelnative = list(item.keys())[0]
                    if 'notes' in item[modelnative].keys():
                        notes = item[modelnative]['notes']
                    else:
                        notes = None
                    native = modelnative.split('|')[1]
        
                elif type(item) is str:
                    modelnative = item
                    native = modelnative.split('|')[1]
        
                try:
                    isos = reg_dic[native]
                    isos = [iso.strip() for iso in isos]
                    isos.sort()
                except(KeyError):
                    isos = []
                # print(yin[0][model][idx])
                new = {modelnative: {'countries': ', '.join(isos)}}
                if notes is not None:
                    new[modelnative]['notes'] = notes
                yin[0][model][idx] = new
                
                idx = idx+1
                notes = None
            with open(f'{folder_native_regions}\\new\\{fn}', 'w') as file:
                documents = yaml.dump(yin, file)
            
        except(FileNotFoundError):
            print(f'SKIPPING {model}, excel not opened')
        
            
        
    
    
#%% Snippet from daniel re OpenEntrance iso3
# import openentrancefrom pathlib import Pathregions = openentrance.RegionCodeList.from_directory(name="region", path=Path("definitions/region/"))current_countries = [regions[openentrance.iso_mapping[c]].iso3 for c in current_list]
    
    
    
    
    
    
    
    
    
    
    
    
    