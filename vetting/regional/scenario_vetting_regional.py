# -*- coding: utf-8 -*-
"""
Vetting script for Climate Advisory Borad - regional version
"""
import os
os.chdir('C:\\Github\\eu-climate-advisory-board-workflow\\vetting')

#%% Import packages and data
# import itertools as it
import time
start = time.time()
print(start)
import numpy as np
import pyam
import pandas as pd
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

meta_docs = {}
historical_columns = []
future_columns = []
value_columns = {}


log = True
modelbymodel = True    # Output single files for each model into Teams folder
single_model = False   # Testing mode- not used
drop_na_yamls = True   #Process only models for which there is a yaml
print_log = print if log else lambda x: None


from vetting_functions import *

# =============================================================================
#%% Configuration
# =============================================================================

#%% Settings for the specific run
region_level = 'regional'
user = 'byers'

ver = 'normal'

# ver = 'ips'

flag_fail = 'Fail'
flag_pass = 'Pass'


config_vetting = f'{region_level}\\config_vetting_{ver}_regional.yaml'
instance = 'eu_climate_submission'

# input_data_ref = f'{region_level}\\input_data\\extra-ref-ar6-201518-data.xlsx'
input_data_ref = f'{region_level}\\input_data\\input_reference_all.csv'
input_yaml_dir = f'..\\definitions\\region\\model_native_regions\\'

"C:/Users/byers/IIASA/ECE.prog - Documents/Projects/EUAB/vetting/regional/input_data/input_reference_edgarCO2CH4.csv",
"C:/Users/byers/IIASA/ECE.prog - Documents/Projects/EUAB/vetting/regional/input_data/input_reference_ieaPE_SE.csv"


# input_data_ceds = f'{region_level}\\input_data\\CEDS_ref_data.xlsx'
input_data_mapping = f'{region_level}\\input_data\\model_region_mapping.csv'
output_folder = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\{region_level}\\output_data\\'

#%% Load data
if modelbymodel==True:
    if not os.path.exists(f'{output_folder}teams'):
        os.makedirs(f'{output_folder}teams')



#%% Settings for the project / dataset
# Specify what data to read in

years = np.arange(2010, 2041, dtype=int).tolist()
year_aggregate = 2020

model = 'MESSAGEix-GLOBIOM 1.1'


varlist = ['Emissions|CO2',
            'Emissions|CO2|Energy and Industrial Processes',
            # 'Emissions|CO2|Energy',
            # 'Emissions|CO2|Industrial Processes',
            # 'Emissions|CO2|AFOLU',
            # 'Emissions|CO2|Other',
            # 'Emissions|CO2|Waste',
            'Emissions|CH4',
            # 'Emissions|N2O',
            'Primary Energy',
            # 'Primary Energy|Fossil',
            # 'Primary Energy|Gas',
            # 'Primary Energy|Oil',
            # 'Primary Energy|Coal',
            'Primary Energy|Nuclear',
            # 'Primary Energy|Solar',
            # 'Primary Energy|Wind',
            # 'Secondary Energy',
            'Secondary Energy|Electricity',
            'Secondary Energy|Electricity|Nuclear',
            'Secondary Energy|Electricity|Solar',
            'Secondary Energy|Electricity|Wind',
            # 'Final Energy',
            # 'GDP|MER',
            # 'Population',
            'Carbon Sequestration|CCS*']



# =============================================================================
#%% Prepare the region-iso mapping dictionaries and aggregate Reference data
# =============================================================================




#% Prepare the region-iso mapping dictionaries

######### (old way using Fabio's excel table)
# reg_mapping_all = pd.read_csv(input_data_mapping)
# reg_mapping = reg_mapping_all[['ISO', model]]
# reg_mapping.rename(columns={model: 'region'}, inplace=True)
# reg_mapping.region.fillna('none', inplace=True)

# reg_mapping = reg_mapping.loc[reg_mapping.region.str.contains('_marker')]
# iso_reg_dict = reg_mapping.groupby('region')['ISO'].apply(list).to_dict()
# unique_natives = iso_reg_dict.keys()


######## New way - with yaml region file
# Get list of yaml filenames
model_yaml_map = pd.read_excel('regional\\input_data\\model_yaml_map.xlsx')
# model_yaml_map.yaml.fillna('none', inplace=True)

# Process only models with a yaml (or not)
if drop_na_yamls:
    model_yaml_map.dropna(subset='yaml', inplace=True)
    model_yaml_map.dropna(subset='include', inplace=True)
    model_yaml_map.set_index('model', inplace=True)
    
    
#%%=============================================================================
# =============================================================================
# # MAJOR LOOP STARTS HERE    
# =============================================================================
# =============================================================================
for model, yamlnm in model_yaml_map.iterrows():
    print(f'################## STARTING {model} #############################')

    reg_yaml = f'{input_yaml_dir}{yamlnm[0]}.yaml'
    
    # Load model region-iso mapping yaml file for model
    with open(reg_yaml, "r") as stream:
        reg_mapping = yaml.safe_load(stream)
    
    # Clean up yaml and into dicts
    reg_mapping = reg_mapping[0]
    yaml_model = list(reg_mapping.keys())[0]
    reg_mapping = reg_mapping[yaml_model]
    reg_mapping = [x for x in reg_mapping if 'Eu' in list(x.keys())[0]]
    reg_mapping = {list(k.keys())[0]:list(list(k.values())[0].values()) for k in reg_mapping }
    iso_reg_dict = {k.split('|')[1]:v for k,v in reg_mapping.items()}
    
    # Split up the ISO values (remove commas etc)
    for k, v in iso_reg_dict.items():
        v = v[0].split(',')
        v = [x.strip() for x in v]
        iso_reg_dict[k] = v
        
    unique_natives = iso_reg_dict.keys()
    
    
    ###############
    # Load reference ISO data
    ref_iso_data = pyam.IamDataFrame(input_data_ref)
    
    #% Aggregate reference data for the model regions
    for native in unique_natives:
        for variable in ref_iso_data.variable:
            ref_iso_data.aggregate_region(variable, region=native, subregions=iso_reg_dict[native], append=True)
    
    
    
    #% Strip marker from region
    # regions = {x :x.split('_')[0] for x in unique_natives}
    # ref_data.rename({'region':regions}, inplace=True)
    
    
    #%% load pyam data for model
    
    regions = [f'{model}|'+x.split('_')[0] for x in unique_natives]
    
    
    dfin = pyam.read_iiasa(instance,
                            model=model,
                            # scenario=scenarios,
                            variable=varlist,
                           year=years,
                            region=regions,
                           )
    
    if len(dfin.region)==0:
        print(f'WARNING - NO NATIVE REGIONS FOR {model}')
    else:
    
        # =============================================================================
        # RESTART from here (without re-loading data from server)
        # =============================================================================
        
        # Inteprolate data
        df = dfin.interpolate(range(years[0], years[-1], 1))
        
        print('loaded')
        print(time.time()-start)
        
        
        
        
        #%%=============================================================================
        
        # Join model with reference data
        agg_region_name = f'{model}|Europe_agg'
        # Aggregate natives
        for variable in ref_iso_data.variable:
            ref_iso_data.aggregate_region(variable, region=agg_region_name, subregions=unique_natives, append=True)
        ref_data = ref_iso_data.filter(region=agg_region_name)
        
        unique_natives = df.region
        for variable in df.variable:
            df.aggregate_region(variable, region=agg_region_name, subregions=unique_natives, append=True)
            
        df = df.filter(region=agg_region_name)
        df = df.append(ref_data)
        
        
        
        #%%
        
        ###################################
        #
        # Set variables and thresholds for
        # each check
        #
        ###################################
        # os.chdir('c://github/ipcc_ar6_scenario_assessment/scripts/vetting/')
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
        # Primary energy - Renewables share, Solar-Wind    UNUSED
        # =============================================================================
        # Renewable share (Primary Energy - (Fossil + Nuclear)) / Primary Energy
        # primary_energy = to_series(df.filter(variable='Primary Energy'))
        # non_renewable = to_series(
        #     df
        #     .filter(variable=['Primary Energy|Fossil', 'Primary Energy|Nuclear'])
        #     .aggregate(variable='Primary Energy')
        # )
        # df.append(
        #     (primary_energy - non_renewable) / primary_energy,
        #     variable='Primary Energy|Renewables share', unit='-',
        #     inplace=True
        # )
        
        # solar_wind = to_series(
        #     df
        #     .filter(variable=['Primary Energy|Solar', 'Primary Energy|Wind'])
        #     .aggregate(variable='Primary Energy')
        # )
        
        # df.append(
        #     solar_wind / primary_energy,
        #     variable='Primary Energy|Solar-Wind share', unit='-',
        #     inplace=True
        # )
        
        # =============================================================================
        ## Secondary energy, electricity
        # =============================================================================
        secondary_energy_electricity = to_series(df.filter(variable='Secondary Energy|Electricity'))
        secondary_wind_solar = to_series(
            df
            .filter(variable=['Secondary Energy|Electricity|Wind', 'Secondary Energy|Electricity|Solar'])
            .aggregate(variable='Secondary Energy|Electricity')
        )
        df.append(
            secondary_wind_solar,
            variable='Secondary Energy|Electricity|Solar-Wind', unit='EJ/yr',
            inplace=True
        )
        df.append(
            secondary_wind_solar / secondary_energy_electricity,
            variable='Secondary Energy|Electricity|Solar-Wind share', unit='-',
            inplace=True
        )
        
        # =============================================================================
        # % increases
        # =============================================================================
        
        # PE Renewable share increase
        # calc_increase_percentage(df, 'Primary Energy|Renewables share', 2020, 2030)
        
        # PE Solar-Wind share increase
        # calc_increase_percentage(df, 'Primary Energy|Solar-Wind share', 2020, 2030)
        
        # SE Solar-Wind share increase
        calc_increase_percentage(df, 'Secondary Energy|Electricity|Solar-Wind', 2020, 2030)
        
        #%%=============================================================================
        # #%% Perform actual checks
        # =============================================================================
        
        ###################################
        #
        # Checks: Aggregates
        #
        ###################################
        
        # Before adding new data, check the aggregates
        
        # =============================================================================
        # Emissions & CCS
        # =============================================================================
        # First, aggregate EIP for CEDS and add it
        # eip = to_series(
        #     df
        #     .filter(variable=['Emissions|CO2|Energy', 'Emissions|CO2|Industrial Processes'], scenario='CEDS')
        #     .aggregate(variable='Emissions|CO2')
        # )
        # df.append(
        #     eip,
        #     variable='Emissions|CO2|Energy and Industrial Processes', unit='Mt CO2/yr',
        #     inplace=True
        # )
        
        # check presence of EIP in other scenarios
        missing = df.require_variable(variable='Emissions|CO2|Energy and Industrial Processes')
        missing = missing.loc[missing.model!='Reference',:]
        
        # if missing, aggregate energy and ip
        mapping = {}
        for model in missing.model.unique():
            mapping[model] = list(missing.loc[missing.model==model, 'scenario'])
        
        # Aggregate and add to the df
        if len(mapping)>0:
            for model, scenarios in mapping.items():
                try:
                    neweip = to_series(
                        df.filter(model=model, scenario=scenarios,
                                  variable=['Emissions|CO2|Energy', 'Emissions|CO2|Industrial Processes'],)
                        .aggregate(variable='Emissions|CO2|Energy and Industrial Processes')
                              )
            
                    df.append(
                        neweip,
                    variable='Emissions|CO2|Energy and Industrial Processes', unit='Mt CO2/yr',
                    inplace=True
                    )
                except(AttributeError):
                    print('No components:{},{}'.format(model, scenarios))
                    pass
        #%
        # Drop the separate components
        df.filter(variable=['Emissions|CO2|Energy', 'Emissions|CO2|Industrial Processes'], keep=False, inplace=True)
        
        #%% First, the aggregation tests ################################
        if aggregation_variables is not None:
            for agg_name, agg_info in aggregation_variables.items():
                filter_check_aggregate(df.filter(year=year_aggregate), 
                                       variable=agg_info['variable'], 
                                       threshold=agg_info['threshold'], 
                                       meta_name=agg_name,
                                       meta_docs=meta_docs,
                                       key_historical=agg_info['key_historical'], 
                                       key_future=agg_info['key_future'], 
                                       ver=ver, 
               )
        else:
            print('Skipping aggregations')
        
        #%%  # Add data: % increases
        # =============================================================================
        
        # Emissions|CO2 increase
        calc_increase_percentage(df, 'Emissions|CO2', 2010, 2020)
        # calc_increase_percentage('Emissions|CO2', 2015, 2020)
        calc_increase_percentage(df, 'Emissions|CO2|Energy and Industrial Processes', 2010, 2020)
        
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
        
        
        
        #%% Second, for the reference checking
        for agg_name, agg_info in reference_variables.items():
            df_ref = create_reference_df(df,
                agg_info['model'], agg_info['scenario'], agg_info['variables_threshold'].keys(), agg_info['ref_year']
            )
            filter_with_reference(df, 
                df_ref, agg_info['variables_threshold'], agg_info['compare_year'], agg_name, meta_docs,
                agg_info['key_historical'], agg_info['key_future'], ver=ver
            )
        
        
        # Third, the bounds tests
        for name, info in bounds_variables.items():
            filter_validate(df, info['variable'], info['year'], info['lo'], info['up'], name,
                info['key_historical'], info['key_future'], info['bound_threshold'], ver=ver
           )
        
        ###################################
        #%% Prepare for excel writeout
        #
        # Create summary columns for historical and future
        #
        ###################################
        
        meta_name_historical = 'Key_historical'
        meta_docs[meta_name_historical] = f'Checks that each of {historical_columns} is a {flag_pass}'
        df.set_meta(flag_fail, name=meta_name_historical)
        
        # NOTE that here we set both "Pass" and ":missing" as PASS (e.e.gmore false psoitives)
        df.meta.loc[
            (df.meta[historical_columns].isin([flag_pass, 'missing'])).all(axis=1),
            meta_name_historical
        ] = flag_pass
        
        meta_name_future = 'Key_future'
        meta_docs[meta_name_future] = f'Checks that each of {future_columns} is a {flag_pass}'
        df.set_meta(flag_fail, name=meta_name_future)
        df.meta.loc[
            (df.meta[future_columns].isin([flag_pass, 'missing'])).all(axis=1),
            meta_name_future
        ] = flag_pass
        
        
        #% Overall - choose that only HISTORICAL == PASS
        col = f'vetting_{ver}'
        # df.meta.loc[(df.meta[meta_name_historical]=='Pass') & (df.meta[meta_name_future]=='Pass'), col] = 'PASS'
        # df.meta.loc[(df.meta[meta_name_historical]=='Fail') | (df.meta[meta_name_future]=='Fail'), col] = 'FAIL'
        df.meta.loc[(df.meta[meta_name_historical]==flag_pass) , col] = 'PASS'
        df.meta.loc[(df.meta[meta_name_historical]==flag_fail) , col] = 'FAIL'
        
        
        
        
        #
        #%% Save to Excel and format output file
        ####################################
        ###################################
        
        
        df.meta['model stripped'] = df.meta.reset_index()['model'].apply(strip_version).values
        
        if modelbymodel==True:
            models = ['all'] + list(df.meta['model stripped'].unique())
        else:
            models = ['all']
        
        for modelo in models:
                modelstr = modelo.replace('/','-')
                xs = '' if modelo=='all' else 'teams\\'
                wbstr = f'{output_folder}{xs}vetting_flags_{modelstr}_{ver}.xlsx'
                writer = pd.ExcelWriter(wbstr, engine='xlsxwriter')
                for x in ['exclude','Source']:
                    try:
                        dfo = df.meta.reset_index().drop(x, axis=1)
                    except(KeyError):
                        pass
                if modelo != 'all':
                    dfo = dfo.loc[dfo['model stripped'].isin([modelo, 'Reference'])]
        
                dfo[['model', 'scenario']+[col, meta_name_historical, meta_name_future]+historical_columns+future_columns].to_excel(writer, sheet_name='vetting_flags', index=False, header=True)
        
                # if ver == 'teams':
                    # dfo = dfo.select_dtypes(object)
        
                cols = dfo.columns.tolist()
                cols = cols[:2] + cols[-1:] +cols[-4:-2] + cols[2:-4]
        
                dfo.to_excel(writer, sheet_name='details', index=False, startrow=1, header=False)
                md = pd.DataFrame(index=meta_docs.keys(), data=meta_docs.values(), columns=['Description'])
                md.to_excel(writer, sheet_name='description')
        
                if ver != 'teams':
                    worksheet = writer.sheets['vetting_flags']
                    worksheet.set_column(0, 0, 13, None)
                    worksheet.set_column(1, 1, 25, None)
                    worksheet.set_column(2, len(dfo.columns)-1, 20, None)
        
        
                # =============================================================================
                #         # Add summary pivot table
                # =============================================================================
        
                dfop = dfo.select_dtypes(object)
                dfop = dfop.loc[dfop.model!='Reference',:]
                cols = dfop.columns[2:-2].tolist() + [col]
                dfop.loc[dfop[col]=='PASS', col] = 'Pass'
                dfop.loc[dfop[col]=='FAIL', col] = 'Fail'
                dfop.rename(index={col: 'OVERALL'}, inplace=True)
        
                # dfp = pd.DataFrame(index=cols, columns=['Pass','Fail','outside_range','missing'])
                # colorder=['Pass','Fail','outside_range','missing']
        
                dfop_simple = dfop[cols].apply(pd.Series.value_counts).sort_index()
                dfop_simple = dfop_simple.T#[colorder]
                dfop_simple.rename(index={col: 'OVERALL'}, inplace=True)
        
        
                if ver != 'teams':
                    dfop_simple.loc[historical_columns,'Key_historical'] = 'Yes'
                    dfop_simple.loc[future_columns,'Key_future'] = 'Yes'
                else:
                    dfop_simple.drop('OVERALL', inplace=True)
        
        
                cols = dfop_simple.columns.tolist()
                try:
                    cols.insert(0, cols.pop(cols.index('Fail')))
                    cols.insert(1, cols.pop(cols.index('Pass')))
        
                except(ValueError):
                    pass
        
        
                # Add Pass+missing column if missing values
                if 'missing' in dfop_simple.columns:
                    Pnkp_name = 'Pass+nonKeyHist_missing'
                    dfop_simple[Pnkp_name] = dfop_simple.Pass
                    # dfop_simple.loc[dfop_simple['Key_historical']!='Yes', Pnkp_name]  =  dfop_simple.loc[dfop_simple['Key_historical']!='Yes', 'Pass']+dfop_simple.loc[dfop_simple['Key historical']!='Yes', 'missing']
                    for row in dfop_simple.itertuples():
                        if (row.Index in historical_columns) and np.isnan(row.missing)==False:
                            dfop_simple.loc[row.Index, Pnkp_name] = row.Pass + row.missing
        
                    cols = dfop_simple.columns.tolist()
                    cols.insert(2, cols.pop(cols.index(Pnkp_name)))
        
                    # dfop_simple[Pnkp_name]
                dfop_simple = dfop_simple.reindex(columns= cols)
                dfop_simple.to_excel(writer, sheet_name='summary_pivot')
        
        
        # =============================================================================
        #         ## Format the detail page
        # =============================================================================
        
                worksheet = writer.sheets['details']
                worksheet.set_column(0, 0, 13, None)
                worksheet.set_column(1, 1, 25, None)
                worksheet.set_column(2, len(dfo.columns)-1, 13, None)
        
                worksheet.freeze_panes(1, 2)
        
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
        
        
        
                for col_num, value in enumerate(dfo.columns.values):
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
        
                # writer.save()
                writer.close()
        
        
        
        ###################################
        #
        #%% Create histograms
        #
        ###################################
        
        # plot_columns = {key: value for key, value in value_columns.items() if value['plot']}
        
        # fig = make_subplots(rows=len(plot_columns), cols=1)
        # color_column = 'model stripped'
        # nbins = 75
        # # fac = 0.5
        # dfo = df.meta.reset_index().drop('exclude', axis=1).copy()
        # try:
        #     colormap = dict(zip(dfo[color_column].unique(), px.colors.qualitative.D3))
        # except:
        #     colormap = None
        
        # in_legend = []
        # threshold_plot = 0.5
        
        # for i, (column, info) in enumerate(plot_columns.items()):
        
        
        #     xmin, xmax = None, None
        #     if info['ref_lo'] is not None and info['ref_up'] is not None:
        #         dx = info['ref_up'] - info['ref_lo']
        #         xmin = info['ref_lo'] - threshold_plot * dx
        #         xmax = info['ref_up'] + threshold_plot * dx
        #     elif info['ref_up'] is not None:
        #         xmax = info['ref_up'] * (1+threshold_plot)
        #     elif info['ref_lo'] is not None and info['unit'] in ['%', '% change']:
        #         xmax = 25 # Purely for visual purposes
        
        #     too_small = dfo[column] <= xmin
        #     too_large = dfo[column] >= xmax
        #     num_too_small = sum(too_small)
        #     num_too_large = sum(too_large)
        
        #     if num_too_small > 0:
        #         dfo.loc[too_small, column] = xmin
        #         fig.add_annotation(
        #             x=xmin, y=num_too_small,
        #             text="≤ x<sub>min</sub>",
        #             showarrow=True, arrowhead=7, ax=20, ay=-20,
        #             col=1, row=i+1
        #         )
        
        #     if num_too_large > 0:
        #         dfo.loc[too_large, column] = xmax
        #         fig.add_annotation(
        #             x=xmax, y=num_too_large,
        #             text="≥ x<sub>max</sub>",
        #             showarrow=True, arrowhead=7, ax=-20, ay=-20,
        #             col=1, row=i+1
        #         )
        
        
        #     # Create a histogram figure, and add all its traces to the main figure
        #     # Set bingroup to None to avoid sharing the same bins accross subplots
        #     for trace in px.histogram(dfo, x=column, color=color_column, nbins=nbins, color_discrete_map=colormap).data:
        #         fig.add_trace(trace.update(
        #             bingroup=None, showlegend=trace.showlegend & (i==0)
        #         ), col=1, row=i+1)
        
        
        #     # fig.update_xaxes(
        #     #     row=i+1, range=[xlo, xup])
        
        #     # Give xaxis its title and change tickformat if necessary
        #     fig.update_xaxes(
        #         row=i+1,
        #         title=column, title_standoff=0,
        #         tickformat='%' if info['unit'] in ['%', '% change'] else None
        #     )
        
        #     # Add reference, lower and upper bounds as lines, if they are defined:
        #     for name, color, label in zip(
        #         ['ref_value', 'ref_lo', 'ref_up'],
        #         ['black', 'mediumseagreen', 'mediumvioletred'],
        #         ['Reference', 'Lower bound', 'Upper bound']
        #     ):
        #         if info[name] is not None:
        #             domain = fig.layout['yaxis{}'.format(i+1)].domain
        #             fig.add_shape(
        #                 type='line',
        #                 xref='x{}'.format(i+1), yref='paper', # yref: stretch full height of plot, independent of y-range
        #                 x0=info[name], x1=info[name], y0=domain[0], y1=domain[1],
        #                 line_color=color
        #             )
        #         # Add legend items, not added when using shapes
        #         if i == 0:
        #             fig.add_scatter(x=[None, None], y=[None, None], name=label,mode='lines', line_color=color)
        
        # fig.update_yaxes(title='# scenarios', title_standoff=0)
        # fig.update_layout(
        #     height=200*len(plot_columns), width=900,
        #     barmode='stack',
        #     margin={'l': 50, 'r': 20, 't': 40, 'b': 60},
        #     legend_y=0.5
        # )
        
        # ##%% Save to html
        # fig.update_layout(width=None).write_html(f'{output_folder}vetting_histograms_'+ver+'.html', include_plotlyjs='cdn')
        print(time.time()-start)
        print(f'############################# FINISHED {model} #############################')
        df.to_excel(f'{output_folder}df_out.xlsx')
    
        