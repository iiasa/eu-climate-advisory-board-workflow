# -*- coding: utf-8 -*-
"""
Regional Vetting script for Climate Advisory Board

Execute this script from within the "vetting" folder
"""
import os
# os.chdir('C:\\Github\\eu-climate-advisory-board-workflow\\vetting')


#%% Import packages and data
import time
start = time.time()
print(start)
import glob
import numpy as np
import pyam
import pandas as pd
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


log = True
modelbymodel = True    # Output single files for each model into Teams folder
single_model = False   # Testing mode- not used
drop_na_yamls = True   # Process only models with the tag Include=True
print_log = print if log else lambda x: None
check_model_regions_in_db = False  # Checks, model by model, for the available regions (based on variable list) (takes time!)
recreate_yaml_region_map = True  # read in excel, and save to yaml
write_out_all = True


from vetting_functions import *

#%% Configuration
# =============================================================================

#%% Settings for the specific run
region_level = 'regional'
datestr = '20230712'


years = np.arange(2010, 2061, dtype=int).tolist()
year_aggregate = 2020

flag_fail = 'Fail'
flag_pass = 'Pass'
flag_pass_missing = 'Pass_missing'
flag_fail_missing = 'Fail_missing'

config_vetting = f'{region_level}\\config_vetting_{region_level}.yaml'
instance = 'eu-climate-advisory-board'


input_data_ref = f'input_data\\input_reference_all.csv'
input_yaml_dir = f'..\\definitions\\region\\model_native_regions\\'
input_yaml_eu_regions = f'..\\definitions\\region\\european-regions.yaml'
fn_yaml_region_map = f'regional\\input_data\\model_yaml_region_map_{datstr}.yaml'

input_data_mapping = f'{region_level}\\input_data\\model_region_mapping.csv'

output_folder = f'outputs\\'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("created folder : ", output_folder)
    
#%% Load data
if not os.path.exists(f'{output_folder}teams'):
    os.makedirs(f'{output_folder}teams')


#%% Settings for the project / dataset
# Specify what data to read in

varlist = ['Emissions|CO2',
            'Emissions|CO2|Energy and Industrial Processes',
            'Emissions|CO2|Energy',
            'Emissions|CO2|Industrial Processes',
            'Emissions|CH4',
            'Emissions|N2O',
            'Primary Energy',
            'Primary Energy|Gas',
            'Primary Energy|Oil',
            'Primary Energy|Coal',
            'Primary Energy|Nuclear',
            'Primary Energy|Solar',
            'Primary Energy|Wind',
            'Secondary Energy|Electricity',
            'Secondary Energy|Electricity|Nuclear',
            'Secondary Energy|Electricity|Solar',
            'Secondary Energy|Electricity|Wind',
            'Carbon Sequestration|CCS*']



#%% Functions

def func_model_iso_reg_dict(yaml_nm='native_iso_EU27'):
    # generate iso_reg_dict based on either yaml filename or based on native_iso for europe

    if yaml_nm[0] == 'native_iso_EU27':
        # Change to import list from nomenclature
        iso_reg_dict = {'EU27': iso_eu27}
        print('Using default EU27')
    else:
        reg_yaml = glob.glob(f'{input_yaml_dir}{yaml_nm}.y*ml')[0]
        print(reg_yaml)
    
        # Load model region-iso mapping yaml file for model
        with open(reg_yaml, "r") as stream:
            reg_mapping = yaml.safe_load(stream)
        
        # Clean up yaml and into dicts
        reg_mapping = reg_mapping[0]
        yaml_model = list(reg_mapping.keys())[0]
        reg_mapping = reg_mapping[yaml_model]
        reg_mapping_eu = reg_mapping #[x for x in reg_mapping if ('Eu' in list(x.keys())[0]) or ('EU' in list(x.keys())[0])]
        
        reg_mapping_eu = {list(k.keys())[0]:list(list(k.values())[0].values()) for k in reg_mapping_eu }
        iso_reg_dict = {k.split('|')[1]:v for k,v in reg_mapping_eu.items()}
        
        # Split up the ISO values (remove commas etc)
        for k, v in iso_reg_dict.items():
            v = v[0].split(',')
            v = [x.strip() for x in v]
            iso_reg_dict[k] = v
        
    all_natives = list(iso_reg_dict.keys())
    return iso_reg_dict, all_natives, yaml_model

#% Prepare for excel writeout
    #
    # Create summary columns for historical and future
    #
    ###################################
def write_out(df, filename, iso_reg_dict={}, model='all', include_data=False, include_meta=False):
    
    '''
    df: pyam.IAMdf to be written out, must have meta
    iso_reg_dict: dictionary of region: ISOs. Only used for reporting
    model: model that is in the data. If 'all', write out all. Reporting is slightly different
    include_data: include the df as timeseries data in output excel
    include_meta: include the metadata in output excel 
    
    
    # =============================================================================
    # Aggregate the Key checks

    # i.e.  historical_columns / future_columns list the indicators marked as "Key"
    # in the config file.
    # First set all scenarios to flag_fail "Fail"
    Then identify all scenarios that have
        1. Pass or missing - and set meta_name_historical column to flag_pass_missing, e.g. Pass_missing
        2. Pass - set to flag_pass e.g. PASS
    # =============================================================================
    '''
    # Historical
    meta_name_historical = 'Key_historical'
    meta_docs[meta_name_historical] = f'Checks that each of {historical_columns} is a {flag_pass}'
    df.set_meta(flag_fail, name=meta_name_historical)
    
    df.meta.loc[
        (df.meta[historical_columns].isin([flag_fail, flag_fail_missing])).all(axis=1),
        meta_name_historical
    ] = flag_fail_missing
    
    # NOTE that here we set both "Pass" and ":missing" as PASS (e.e.gmore false psoitives)

    # First Set rows with all pass or missing to pass_missing (must be this order)
    df.meta.loc[
        (df.meta[historical_columns].isin([flag_pass, flag_pass_missing])).all(axis=1),
        meta_name_historical
    ] = flag_pass_missing

    # Now set only those with all pass to pass
    df.meta.loc[
        (df.meta[historical_columns].isin([flag_pass,])).all(axis=1),
        meta_name_historical
    ] = flag_pass
    
    # Future
    meta_name_future = 'Key_future'
    meta_docs[meta_name_future] = f'Checks that each of {future_columns} is a {flag_pass}'
    df.set_meta(flag_fail, name=meta_name_future)
    
    df.meta.loc[
        (df.meta[future_columns].isin([flag_fail, flag_fail_missing])).all(axis=1),
        meta_name_future
    ] = flag_fail_missing
    
    # First Set rows with all pass or missing to pass_missing (must be this order)
    df.meta.loc[
        (df.meta[future_columns].isin([flag_pass, flag_pass_missing])).all(axis=1),
        meta_name_future
    ] = flag_pass_missing
 
    # Now set only those with all pass to pass
    df.meta.loc[
        (df.meta[future_columns].isin([flag_pass, ])).all(axis=1),
        meta_name_future
    ] = flag_pass
    
    
    # Count # of missings in each row
    df.meta[flag_pass_missing] = df.meta.apply(pd.value_counts, axis=1)[flag_pass_missing]
    df.meta[flag_fail_missing] = df.meta.apply(pd.value_counts, axis=1)[flag_fail_missing]
    
    #% OVERALL - choose that only HISTORICAL == PASS
    col = f'vetting_{region_level}'
    df.meta.loc[(df.meta[meta_name_historical]==flag_pass) , col] = 'PASS'
    df.meta.loc[(df.meta[meta_name_historical]==flag_pass_missing) , col] = flag_pass_missing
    df.meta.loc[(df.meta[meta_name_historical]==flag_fail) , col] = 'FAIL'
    
    
    #% Save to Excel and format output file
    ####################################
    ###################################
    
    df.meta['model_stripped'] = df.meta.reset_index()['model'].apply(strip_version).values

        
    modelstr = model.replace('/','-')
    xs = '' if model=='all' else 'teams\\'
    modelstr = 'all' if model=='all' else modelstr
    wbstr = f'{output_folder}{xs}vetting_flags_{modelstr}_{region_level}_{datestr}.xlsx'
    writer = pd.ExcelWriter(wbstr, engine='xlsxwriter')
    
    # Strip out exclude / source columns if present
    dfo = df.meta.reset_index()
    covcols = [x  for x in dfo.columns if 'coverage' in x]
    for x in ['exclude','source','Source','version','doi','reference']+covcols:
            dfo.drop(x, axis=1, errors='ignore', inplace=True)
        
    if model != 'all':
        dfo = dfo.loc[dfo['model'].isin([model, 'Reference'])]

    # =============================================================================
    #     # Write vetting flags sheet
    # =============================================================================
    vetting_cols = ['model', 'model_stripped', 'scenario'] \
            + [ col,'vetted_type', meta_name_historical, meta_name_future, flag_pass_missing, flag_fail_missing] \
            + historical_columns \
            + future_columns 
            
    dfo[vetting_cols].to_excel(writer, sheet_name='vetting_flags', index=False, header=True)

    # =============================================================================
    #     # Write details sheet
    # =============================================================================
    cols1 = dfo.columns.tolist()
    detail_cols = vetting_cols[:8]  + cols1[3:-5] #bring the key vetted columns to front + 
    
    
    # Don't include details sheet in all, because too many different columns
    dfo[detail_cols].to_excel(writer, sheet_name='details', index=False, startrow=1, header=False)
    
    
    md = pd.DataFrame(index=meta_docs.keys(), data=meta_docs.values(), columns=['Description'])
    md.to_excel(writer, sheet_name='description')


    # =============================================================================
    #       # Model summary / detail pivot tables
    # =============================================================================
    if model == 'all':
        cols = ['model',  col, 'Key_historical','Key_future','IEA Primary Energy summary','IEA Electricity summary','EEA GHG 2021 summary'] #'scenario',
        dfom = dfo.copy(deep=True)[cols]
        dfom = dfom.loc[dfom.model!='Reference',:]
        dfom.fillna(flag_pass_missing, inplace=True)
        
        ao = dfom.groupby(['model',col]).value_counts()
        ao.to_excel(writer, sheet_name='model_detail')

        for c in cols[2:]:
            dfom.loc[dfom[c]=='Pass', c] = 1
            dfom.loc[dfom[c]=='Fail', c] = 0
            dfom.loc[dfom[c]==flag_pass_missing, c] = 0

        dfom = dfom.groupby(['model',col]).sum() 
        dfom.groupby(['model',col]).value_counts()
       
        dfom.to_excel(writer, sheet_name='model_summary')

    # =============================================================================
    #         # Add summary pivot table
    # =============================================================================

    dfop = dfo.select_dtypes(object)
    dfop = dfop.loc[dfop.model!='Reference',:]
    dfop.loc[dfop[col]=='PASS', col] = 'Pass'
    dfop.loc[dfop[col]=='FAIL', col] = 'Fail'
    dfop.rename(columns={col: 'OVERALL'}, inplace=True)

    cols = dfop.columns[3:-1]#historical_columns + future_columns + dfop.columns[5:7].tolist() + dfop.columns[3:5].tolist() 

    dfop_simple = dfop[cols].apply(pd.Series.value_counts).fillna(0).sort_index()
    dfop_simple = dfop_simple.T
    dfop_simple.rename(index={col: 'OVERALL'}, inplace=True)


    if region_level != 'teams':
        dfop_simple.loc[historical_columns,'Key_historical_check'] = 'Yes'
        dfop_simple.loc[future_columns,'Key_future_check'] = 'Yes'


    cols = dfop_simple.columns.tolist()
    try:
        cols.insert(0, cols.pop(cols.index('Fail')))
        cols.insert(1, cols.pop(cols.index('Pass')))
    except(ValueError):
        pass

    dfop_simple = dfop_simple.reindex(columns= cols)
    dfop_simple['sum_check'] = dfop_simple.sum(axis=1)
    dfop_simple.to_excel(writer, sheet_name='summary_pivot')

    # Add regions dictionary
    try:
        irdout = pd.concat([pd.DataFrame(v, columns=[k]) for k, v in iso_reg_dict.items()], axis=1)
        pd.DataFrame(irdout).to_excel(writer, sheet_name='region_mapping')
    except(ValueError):
        print('Could not write region mapping')
    # Add data
    if include_data:
        df.to_excel(writer, sheet_name='data', include_meta=include_meta,)

# =============================================================================
#   ## Format the pages 
# =============================================================================

    # vetting_flags page
    worksheet = writer.sheets['vetting_flags']
    worksheet.set_column(0, 0, 13, None)
    worksheet.set_column(1, 1, 25, None)
    worksheet.set_column(2, len(dfo.columns)-1, 20, None)
    worksheet.freeze_panes(1, 3)
    worksheet.autofilter(0, 0, len(dfo), len(dfo.columns)-1)

    
    # Details page
    worksheet = writer.sheets['details']
    worksheet.set_column(0, 0, 13, None)
    worksheet.set_column(1, 1, 25, None)
    worksheet.set_column(2, len(dfo.columns)-1, 13, None)
    worksheet.freeze_panes(1, 3)

    # Write header manually with custom style for text wrap
    workbook = writer.book
    header_format_creator = lambda is_bold: workbook.add_format({
        'bold': is_bold,
        'text_wrap': True,
        'align': 'center',
        'valign': 'top',
        'border': 1
    })
    header_format = header_format_creator(True)
    subheader_format = header_format_creator(False)

    for col_num, value in enumerate(detail_cols):
        curr_format = subheader_format if value[0] == '(' or value[-1] == ']' else header_format
        worksheet.write(0, col_num, value, curr_format)

    # Add autofilter
    worksheet.autofilter(0, 0, len(dfo), len(dfo.columns)-1)

    # Change format of value columns
    largenum_format = workbook.add_format({'num_format': '0'})
    percentage_format = workbook.add_format({'num_format': '0%'})
    percentage_change_format = workbook.add_format({'num_format': '+0%;-0%;0%'})
    for i, column in enumerate(dfo.columns):
        unit = value_columns.get(column, {}).get('unit', None)
        if unit == '%':
            worksheet.set_column(i, i, None, percentage_format)
        if unit == '% change':
            worksheet.set_column(i, i, None, percentage_change_format)
        if dfo[column].dtype == float and dfo[column].median() > 10:
            worksheet.set_column(i, i, None, largenum_format)
    
    writer.sheets['description'].set_column(0, 1, 35, None)

    # Fromat summary pivot page
    writer.sheets['summary_pivot'].set_column(0, 0, 60, None)
    writer.sheets['summary_pivot'].set_column(1, 6, 20, None)


    writer.close()
    return wbstr

# ================================================================
#%% Prepare the region-iso mapping dictionaries and aggregate Reference data
# =============================================================================
#% Prepare the region-iso mapping dictionaries

# =============================================================================
# Optional - only needs to done once at beginning
# =============================================================================
# Read in excel, convert to dict, write out to yaml (to make mapping yaml first time)
if recreate_yaml_region_map:
    model_yaml_map = pd.read_excel('regional\\input_data\\model_yaml_region_map.xlsx')
    model_yaml_map.set_index('model', inplace=True,)
    model_yaml_map.sort_index(inplace=True)
    model_yaml_map.dropna(axis=1, how='all', inplace=True)
    model_yaml_map.yaml.fillna('None', inplace=True)
    
    with open(fn_yaml_region_map, 'w') as file:
          documents = yaml.dump(model_yaml_map.to_dict(orient='index') , file)


# Load the yaml file that specifies:
#    - which yaml file to read from definitions for the native-iso mapping
#    - which common or native or iso regions to use for regional vetting


# Load yaml file
with open(fn_yaml_region_map, 'r') as file:
     model_yaml_map = yaml.safe_load(file)    
model_yaml_map = pd.DataFrame(model_yaml_map).T  

# Process only models with a yaml native-iso mapping (or not)
if drop_na_yamls:
    # model_yaml_map.dropna(subset='yaml', inplace=True)
    model_yaml_map = model_yaml_map[model_yaml_map.include != False]
    
    model_yaml_map.dropna(subset=['vetted_regions', 'vetted_type'], how='any', inplace=True)
    

# =============================================================================
# Optional - Query model regions available (to update yaml / excels above)
# =============================================================================
# Only do this if checking the regions available for each model and updating the model_yaml_region_map
if check_model_regions_in_db:
    regions_available = {}
    for model in model_yaml_map.iterrows():
        print(model)
        dfin = pyam.read_iiasa(instance,
                                model=model[0],
                                # scenario=scenarios,
                                variable=varlist,
                                year=years)
        
        regions_available[model[0]] = dfin.region
        
        
# =============================================================================
# Load late submissions
# =============================================================================
# if load_late_submissions:
    
#     dfinlate = pyam.read_iiasa(instance,
#                             model='lalala',
#                             # scenario=scenarios,
#                             variable=varlist,
#                             year=years)
    
#     # Load late submissions scenarios
#     for f in glob.glob(f'{late_submissions_path}*late-submission*.xlsx'):
#         dft = pyam.IamDataFrame(f)
#         dfinlate.append(dft, inplace=True)
        
    
#%%=============================================================================
# =============================================================================
# # MAJOR LOOP STARTS HERE    
# =============================================================================
# =============================================================================

# Load EU definitions

with open(input_yaml_eu_regions , 'r') as file:
     eu_regions = yaml.safe_load(file)[0]['Europe']
     
     iso_eu27 = [x for x in eu_regions if 'EU27' in x.keys()][0]['EU27']['countries']
     iso_eu27 = iso_eu27.split(', ')

iso_eu28 = iso_eu27 + ['GBR']
iso_euMC = [x for x in iso_eu27 if x not in ['CYP','MLT','GBR']]
iso_eurR10 = ['ALB', 'AND', 'AUT', 'BLR', 'BEL', 'BIH', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FRO', 'FIN', 'FRA', 'DEU', 'GIB',
 'GRC', 'VAT', 'HUN', 'ISL', 'IRL', 'ITA', 'XKX', 'LVA', 'LIE', 'LTU', 'LUX', 'MLT', 'MDA', 'MCO', 'MNE', 'NLD', 'MKD', 'NOR', 'POL', 'PRT', 'ROU', 'SMR', 'SRB', 'SVK', 'SVN', 'ESP', 'SWE', 'CHE', 'GBR', 'UKR']

allowed_common_regions = {'EU27': iso_eu27,
                          'EU27 & UK': iso_eu28,
                          'EU27 (excl. Malta & Cyprus)': iso_euMC,
                          'Europe (R10)': iso_eurR10,
                          }

iso_reg_dict_all = {}

ct = 0

for model, attr in model_yaml_map.iloc[:].iterrows(): #.iloc[:4]
    print(f'################## STARTING {model} #############################')


    # attr.vetted_regions - this is the selected region(s) for the model
    # iso_reg_dict - this is the dict made to map the ISOs to the native/common region used for aggregation
    # agg_region_name - this is the name of the comparison region, for both model (df) and reference (ref_data) dataframes
        
    # Use cases
    
    # Common region    
    if (attr.vetted_type == 'common') & (attr.vetted_regions in allowed_common_regions.keys()):
        

        # ref region: e.g. 'EU27'
        # model region: e.g. EU27
        
        # get list of isos in common region
        ref_isos = allowed_common_regions[attr.vetted_regions]
        iso_reg_dict = {attr.vetted_regions: ref_isos}
        sel_natives = list(iso_reg_dict.keys())
        
        regions = attr.vetted_regions.split(',')
        agg_region_name = regions[0]

    # Native region       
    elif (attr.vetted_type == 'native'):
        
        # ref region: '{model}|Europe_agg'
        # model region: '{model}|Europe_agg'
        
        # Load and generate region - iso dictionary for specific model
        iso_reg_dict, all_natives, yaml_model = func_model_iso_reg_dict(yaml_nm=attr.yaml)
        regions = attr.vetted_regions.split(',')

        sel_natives = [x.split('|')[1] for x in regions]
        iso_reg_dict = {key: iso_reg_dict[key] for key in sel_natives}
        
        agg_region_name = f'{model}|Europe_agg'

    # Error
    else:
        print('Skipping {model}.....')
        continue
        
    
    # Apply to both: add model / regions / isos to big dictionary.
    # This is written to excel/yaml at the end.
    iso_reg_dict_all[model] = iso_reg_dict
    
    ###############
    # Load reference ISO data (e.g. EDGAR, EEA, IEA, EMBERS)
    ref_iso_data = pyam.IamDataFrame(input_data_ref)
    rs = [x for l in iso_reg_dict.values() for x in l]
    ref_iso_data.filter(region=rs, inplace=True)
    
    # Rename EEA Emissions|CH4 variable
    ref_iso_data.rename({'variable':{'Emissions|CH4 (EEA)': 'Emissions|CH4'}}, inplace=True)
    
    if ct==0:
        dfall = ref_iso_data.filter(model='xxx') # create empty IamDF for saving all data
    
    #% Aggregate reference data for the model regions
    for native in sel_natives:
        for variable in ref_iso_data.variable:
            # print(variable)
            ref_iso_data.aggregate_region(variable, region=native, subregions=iso_reg_dict[native], append=True)
    
    
    #%% load pyam data for model
    dfin = pyam.read_iiasa(instance,
                            model=model,
                            # scenario=scenarios,
                            variable=varlist,
                            year=years,
                            region=regions,
                            meta=False
                           )
    
    
    
    if len(dfin.region)==0:
        print(f'WARNING - NO NATIVE REGIONS FOR {model}')
        continue
    # else:

    # =============================================================================
    # RESTART from here (without re-loading data from server)
    # =============================================================================
    
    # Inteprolate data
    dfin_ymin =  np.min(dfin.year) if np.min(dfin.year) > years[0] else years[0]
    dfin_ymax =  np.max(dfin.year) if np.max(dfin.year) < years[-1] else years[-1]

    try:
        df = dfin.interpolate(range(dfin_ymin, dfin_ymax+1, 1))
    except(ValueError):
        dft = dfin.timeseries().fillna(0)
        dfin = pyam.IamDataFrame(dft)
        df = dfin.interpolate(range(dfin_ymin, dfin_ymax+1, 1))

    print('loaded')
    print(time.time()-start)
    
    
    #%%=============================================================================
    
    # If native regions, 
    # Aggregate the native reference and model data
    if attr.vetted_type == 'native':
        # Aggregate natives regions to synthetic comparison region
        for variable in ref_iso_data.variable:
            ref_iso_data.aggregate_region(variable, region=agg_region_name,
                                          subregions=sel_natives, append=True)
            
        # aggregate the model data to the common region
        for variable in df.variable:
            df.aggregate_region(variable, region=agg_region_name, subregions=regions, append=True)    
    
    
    
    # Filter data to the desired comparison region
    ref_data = ref_iso_data.filter(region=agg_region_name)
    df = df.filter(region=agg_region_name)

   #%% 
    if len(ref_data)==0:
        print('ERROR - no {agg_region_name} reference data for {model}')
        continue
    if len(df)==0:
        print('ERROR - no {agg_region_name} data for {model}')
        continue
    
    #%%
    # Join model with reference data
    df = df.append(ref_data)
    df.set_meta(attr.vetted_type,'vetted_type')
    
    ###################################
    # Load config and Set variables and thresholds for
    # each check
    ###################################
    # read from local folder
    with open(config_vetting, 'r', encoding='utf8') as config_yaml:
        config = yaml.safe_load(config_yaml)
    reference_variables = config['reference_variables']
    aggregation_variables = config['aggregation_variables']
    bounds_variables = config['bounds_variables']
    
    if single_model:
        df = df.filter(model=['MESSAGE*', 'Reference'])  # Choose one arbitrary model, and the Reference data
    
    #%% Create additional variables
    
    ###################################
    #
    # Create additional variables
    #
    ###################################
    
    
    # =============================================================================
    ## Secondary energy, electricity
    # =============================================================================
    secondary_energy_electricity = to_series(df.filter(variable='Secondary Energy|Electricity'))
    # Aggregate to new wind-solar variable
    swv = 'Secondary Energy|Electricity|Solar-Wind'
    secondary_wind_solar = df.add('Secondary Energy|Electricity|Wind',
                                  'Secondary Energy|Electricity|Solar',
                                  swv)
    
    df.append(
        secondary_wind_solar.filter(model='Ref*', keep=False),
        inplace=True
    )
    
    # Calculate Share of wind-solar of total electricity
    swvs = 'Secondary Energy|Electricity|Solar-Wind share'
    df.divide(swv, 'Secondary Energy|Electricity', swvs,
              ignore_units='-', append=True)
    
    
    # =============================================================================
    # % increases
    # =============================================================================
    
    # SE Solar-Wind share increase
    calc_increase_percentage(df, 'Secondary Energy|Electricity|Solar-Wind', 2020, 2030)
    
    #%=============================================================================
    # #%% Perform actual checks
    # =============================================================================
    
    ###################################
    #
    # Checks: Aggregates
    #
    ###################################
    
    
    # check presence of EIP in the scenarios (not Reference)
    missing = df.require_variable(variable='Emissions|CO2|Energy and Industrial Processes')
    if missing is not None:
        missing = missing.loc[missing.model!='Reference',:]
        components = ['Emissions|CO2|Energy', 'Emissions|CO2|Industrial Processes']
        # if missing, aggregate energy and ip
        mapping = {}
        for model in missing.model.unique():
            mapping[model] = list(missing.loc[missing.model==model, 'scenario'])
        
        # Aggregate and add to the df
        if len(mapping)>0:
            for model, scenarios in mapping.items():
                try:
                    neweip = to_series(
                        df.filter(model=model, scenario=scenarios, variable=components).aggregate(variable='Emissions|CO2|Energy and Industrial Processes', components=components)
                        )
            
                    df.append(
                        neweip,
                    variable='Emissions|CO2|Energy and Industrial Processes', unit='Mt CO2/yr',
                    inplace=True
                    )
                except(IndexError):
                    print('No components:{},{}'.format(model, scenarios))
        #%
        # Drop the separate components
    
    #%% First, the aggregation tests ################################
    
    meta_docs = {}
    historical_columns = []
    future_columns = []
    value_columns = {}    
    
    
    if aggregation_variables is not None:
        for agg_name, agg_info in aggregation_variables.items():
            filter_check_aggregate(df.filter(year=year_aggregate), 
                                   variable=agg_info['variable'], 
                                   threshold=agg_info['threshold'], 
                                   meta_name=agg_name,
                                   meta_docs=meta_docs,
                                   key_historical=agg_info['key_historical'], 
                                   key_future=agg_info['key_future'],
                                   historical_columns=historical_columns,
                                   future_columns=historical_columns,
                                   ver=region_level, 
           )
    else:
        print('Skipping aggregations')

    # =============================================================================
    #%  # Add data: % increases
    # =============================================================================
    
    # Emissions|CO2 increase
    for v in ['Emissions|CO2|Energy and Industrial Processes',
              'Emissions|CO2|Energy']:
           calc_increase_percentage(df, v, 2015, 2020)


    # Calculate CCS from energy (not industrial):
    try:
        df.append(
            to_series(
                df
                .filter(variable=['Carbon Sequestration|CCS|Biomass', 'Carbon Sequestration|CCS|Fossil'])
                .aggregate(variable='Carbon Sequestration|CCS')
            ),
            variable='Carbon Sequestration|CCS|Biomass and Fossil', unit='Mt CO2/yr',
            inplace=True
        )
    except(AttributeError):
        print('Skip CCS aggregation for {model}')
    
    
    # =============================================================================
    #  #% Do checks against reference data
    # =============================================================================
    for agg_name, agg_info in reference_variables.items():
        df_ref = create_reference_df(df,
            agg_info['model'], agg_info['scenario'], agg_info['variables_threshold'].keys(), agg_info['ref_year']
        )
        filter_with_reference(df, 
                              df_ref, 
                              agg_info['variables_threshold'],
                              agg_info['missing_flags'],
                              agg_info['compare_year'], 
                              agg_name, 
                              meta_docs,
                              agg_info['key_historical'], 
                              agg_info['key_future'], 
                              historical_columns, 
                              future_columns, 
                              value_columns=value_columns,
                              ver=region_level,
        )
    
    
    # =============================================================================
    # Do the bounds checks
    # =============================================================================
    for name, info in bounds_variables.items():
        filter_validate(df, 
                        info['variable'], 
                        info['missing_flag'],
                        info['year'], 
                        info['lo'], 
                        info['up'], 
                        name,
                        info['key_historical'], 
                        info['key_future'], 
                        historical_columns, 
                        future_columns,     
                        info['bound_threshold'], 
                        ver=region_level,
       )
    
    ###################################
    
    dfall.append(df.filter(model=model), inplace=True)
    try:
        if ct==0:
            dfall.append(df.filter(model='Reference'), 
                     ignore_meta_conflict=True,
                     inplace=True)
    except(ValueError):
        print('Ref data already present')
        
    print(f'############  Finished {model} ############')
    
#%%
    # Write out to excel
    write_out(df, iso_reg_dict, model=model, include_data=True, include_meta=False)
    ct=ct+1
print('Finished loop, writing out all')
if write_out_all:
    fn = write_out(dfall, iso_reg_dict_all, model='all', include_data=True, include_meta=False)
    os.startfile(fn)
    
    
#%% Write out full dictionary of regions used
for mk, mv in iso_reg_dict_all.items():
    for k, v in mv.items():
        v = ', '.join(v)
        iso_reg_dict_all[mk][k] = v

with open(f'{output_folder}\\model_reg_iso_output.yaml', 'w') as file:
     documents = yaml.dump(iso_reg_dict_all , file)
     