# -*- coding: utf-8 -*-
"""
Vetting script for Climate Advisory Borad
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


# =============================================================================
#%% Configuration
# =============================================================================

#%% Settings for the specific run

log = True
modelbymodel = True
single_model = False
ver = 'normal'
# ver = 'ips'

flag_fail = 'Fail'
flag_pass = 'Pass'


config_vetting = f'config_vetting_{ver}.yaml'
instance = 'eu_climate_submission'

input_data_ref = f'input_data\\extra-ref-ar6-201518-data.xlsx'
input_data_ceds = f'input_data\\CEDS_ref_data.xlsx'
output_folder = 'output_data\\'

#%% Load data
if modelbymodel==True:
    if not os.path.exists(f'{output_folder}teams'):
        os.makedirs(f'{output_folder}teams')



#%% Settings for the project / dataset
# Specify what data to read in

years = np.arange(2000, 2101, dtype=int)

# models = 'MESSAGE*'
# scenarios = '*'


varlist = ['Emissions|CO2',
            'Emissions|CO2|Energy and Industrial Processes',
            # 'Emissions|CO2|Energy', 
            'Emissions|CO2|Industrial Processes',
            # 'Emissions|CO2|AFOLU',
            # 'Emissions|CO2|Other',
            # 'Emissions|CO2|Waste',
            'Emissions|CH4',
            'Emissions|N2O',
            'Primary Energy',
            'Primary Energy|Fossil',
            'Primary Energy|Nuclear',
            'Primary Energy|Solar',
            'Primary Energy|Wind',            
            # 'Secondary Energy',
            'Secondary Energy|Electricity',
            'Secondary Energy|Electricity|Nuclear',
            'Secondary Energy|Electricity|Solar',
            'Secondary Energy|Electricity|Wind',
            # 'Final Energy',
            # 'GDP|MER',
            # 'Population',
            'Carbon Sequestration|CCS*']

region = ['World']

#%% load pyam data

dfin = pyam.read_iiasa(instance, 
                        # model=models, 
                        # scenario=scenarios, 
                       variable=varlist, 
                       year=years,
                       region=region)


print('loaded')
print(time.time()-start)


#%%=============================================================================
# #%% Restart from here if necessary
# =============================================================================

# Drop unwanted scenarios
# manual_scenario_remove(df, remdic)

# Inteprolate data
df = dfin.interpolate(range(years[0], years[-1], 1))

meta_docs = {}
historical_columns = []
future_columns = []
value_columns = {}


print_log = print if log else lambda x: None


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
    df = df.filter(model=['REMIND*', 'Reference'])  # Choose one arbitrary model, and the Reference data

#%% Define functions used to perform checks

###################################
#
# Define functions used to perform
# checks
#
###################################

def create_reference_df(ref_model, ref_scenario, check_variables, ref_year):
    return (
        df
        .filter(model=ref_model, scenario=ref_scenario, variable=check_variables, year=ref_year)
        .timeseries()
        [ref_year]
    )


## Some utilities for the filter functions:
#  - Initialise the meta column
#  - Perform check if a variable exists
#  - Set corresponding scenarios to `fail` given an output of *.validate() or *.check_aggregate

def util_filter_init(meta_name, meta_doc, key_historical, key_future):
    """Creates the meta column"""
    meta_docs[meta_name] = meta_doc
    df.set_meta(flag_pass, name=meta_name)
    df.reset_exclude()
    if key_historical==True:
        historical_columns.append(meta_name)
    if key_future==True:
        future_columns.append(meta_name)

def util_filter_check_exists(meta_name, variable, year=None):
    var_doesnt_exist = pyam.require_variable(df, variable=variable, year=year, exclude_on_fail=True)
    if var_doesnt_exist is not None:
        missing_indices = var_doesnt_exist[['model', 'scenario']]
        df.set_meta('missing', meta_name, missing_indices)

def util_filter_set_failed(meta_name, failed_df, label='Fail'):
    if failed_df is None:
        print_log(f'All scenarios passed validation {meta_name}')
    else:
        print_log('{} scenarios did not pass validation of {}'.format(len(failed_df), meta_name))
        
        # depending on pyam version, if failed_df is a series (new pyam >=0.7), 
        if type(failed_df)==pd.Series:
            failed_indices = (
                failed_df
                # .drop('variable',)
                .reset_index()
                [['model', 'scenario']]
                .drop_duplicates()
            )
        else: #or a DF (older pyam <0.7)
            failed_indices = (
                failed_df
                .drop('variable', axis=1)
                .reset_index()
                [['model', 'scenario']]
                .drop_duplicates()
            )
        
        
        df.set_meta(label, meta_name, failed_indices)
        df.set_meta(True, 'exclude', failed_indices)

def util_get_unit(variable):
    variables = df.as_pandas()[['variable','unit']].drop_duplicates()
    unit = variables[variables['variable'] == variable]['unit']
    try:
        return unit.iloc[0]
    except:
        return 'Unit not found'

def util_filter_save_value_column(name, variable_or_values, year=None, method=None, ref_value=None, ref_lo=None, ref_up=None, plot=True, **kwargs):
    if type(variable_or_values) == str:
        unit = util_get_unit(variable_or_values)
        column_name = f'{name} (model value in {year}) [{unit}]'
        df.set_meta_from_data(name=column_name, variable=variable_or_values, year=year, method=method, **kwargs)
    else:
        unit = util_get_unit(variable_or_values.reset_index()['variable'].unique()[0])
        column_name = f'{name}' # (value)
        df.set_meta(variable_or_values, name=name, **kwargs)
    value_columns[column_name] = {
        'unit': unit,
        'ref_value': ref_value,
        'ref_lo': ref_lo,
        'ref_up': ref_up,
        'plot': plot
    }

## The filter functions:
#  - Filter with reference: compare to a reference value
#  - Filter check aggregate: check if components of variable sum up to main component
#  - Filter validate: check if variable is within bounds

def filter_with_reference(ref_df, thresholds, year, meta_name, key_historical=True, key_future=False, flag_fail='Fail', flag_pass='Pass'):
    meta_name_agg = f'{meta_name} summary'
    trs = '' if ver=='teams' else f': {thresholds}'
    util_filter_init(meta_name_agg, f'Checks that scenario is within reasonable range'+trs,
                     key_historical, key_future)

    curr_meta_columns = []
    # Loop over each variable and value
    for idx, val in ref_df.iteritems():
        _, _, curr_region, curr_variable, curr_unit, *_ = idx

        valstr = np.round(val, 2)
        curr_meta_column = f'({meta_name} {year}: {curr_variable} = {valstr})'
        curr_meta_columns.append(curr_meta_column)
        df.set_meta(flag_pass, curr_meta_column)

        # Step 1: Identify scenarios where this variable/year is missing
        util_filter_check_exists(curr_meta_column, curr_variable, year)

        # Step 2: Identify scenarios where the value is out of range
        lo = val * (1-thresholds[curr_variable])
        up = val * (1+thresholds[curr_variable])
        util_filter_save_value_column(
            curr_meta_column, curr_variable, year=year,
            ref_value=val, ref_lo=lo, ref_up=up
        )
        outside_range = df.filter(
            year=year, region=curr_region
        ).validate(
            criteria={ curr_variable: {
                'lo': lo,
                'up': up
            }},
            exclude_on_fail=True
        )

        util_filter_set_failed(curr_meta_column, outside_range, label='outside_range')

    # df.set_meta(flag_fail, name=meta_name_agg, index=df.filter(exclude=True))
    # df.set_meta(flag_pass, name=meta_name_agg, index=df.filter(exclude=False))
    df.set_meta(flag_fail, name=meta_name_agg)
    df.meta.loc[
        df.meta[curr_meta_columns].isin([flag_pass, 'missing']).all(axis=1),
        meta_name_agg] = flag_pass
    
    

    df.reset_exclude()


def filter_check_aggregate(variable, threshold, meta_name, key_historical=True, key_future=False):
    meta_name_agg = f'{meta_name}_aggregate'
    trs = 'approximately the' if ver=='teams' else ' within {:.0%} of'.format(threshold)
    meta_doc = 'Checks that the components of {} sum up to {} aggregate'.format(
        variable, trs)
    util_filter_init(meta_name_agg, meta_doc, key_historical, key_future)

    failed = df.check_aggregate(variable=variable, rtol=threshold)
    if failed is not None:
        max_rel_difference = ((failed['variable'] - failed['components']) / failed['variable']).max(level=[0,1,2,3])
        util_filter_save_value_column(meta_name_agg, max_rel_difference, plot=False)

    util_filter_set_failed(meta_name_agg, failed)
    df.reset_exclude()

def filter_validate(variable, year, lo, up, meta_name, key_historical=True, key_future=False, 
                    bound_threshold=1):

    if bound_threshold != 1:
        lo = lo * (1-bound_threshold)
        up = up * (1+bound_threshold)
    
    
    if type(year) == str:
        # Interpret string as range:
        year = range(*[int(x) for x in year.split('-')])

    meta_name = f'{meta_name} validate'
    trs = 'in {}'.format(year) if ver=='teams' else 'are within {} and {} in {}'.format(
        variable, lo, up, 'any year' if year is None else year
    )
    meta_doc = f'Checks that the values of {trs} '
    util_filter_init(meta_name, meta_doc, key_historical, key_future)
    util_filter_check_exists(meta_name, variable)

    if type(year) == range:
        if lo is not None:
            util_filter_save_value_column(f'{variable} min', variable, year=year, method=np.min, ref_lo=lo)
        if up is not None:
            util_filter_save_value_column(f'{variable} max', variable, year=year, method=np.max, ref_up=up)
    else:
        util_filter_save_value_column(variable, variable, year=year, ref_lo=lo, ref_up=up)
    failed = df.filter(year=year).validate({variable: {'lo': lo, 'up': up}})

    util_filter_set_failed(meta_name, failed)
    
    df.reset_exclude()

#%% Create additional variables

###################################
#
# Create additional variables
#
###################################

def to_series(pyam_df):
    cols = pyam.utils.YEAR_IDX
    return (
        pyam_df
        .data[cols+['value']]
        .set_index(cols)
        .value
    )


# =============================================================================
# Add additional reference data
# =============================================================================
dfref = pyam.IamDataFrame(input_data_ref)
dfceds =  pyam.IamDataFrame(input_data_ceds)
df.append(dfref, inplace=True)
df.append(dfceds, inplace=True)



# =============================================================================
# Primary energy
# =============================================================================
# Renewable share (Primary Energy - (Fossil + Nuclear)) / Primary Energy
primary_energy = to_series(df.filter(variable='Primary Energy'))
non_renewable = to_series(
    df
    .filter(variable=['Primary Energy|Fossil', 'Primary Energy|Nuclear'])
    .aggregate(variable='Primary Energy')
)
df.append(
    (primary_energy - non_renewable) / primary_energy,
    variable='Primary Energy|Renewables share', unit='-',
    inplace=True
)
solar_wind = to_series(
    df
    .filter(variable=['Primary Energy|Solar', 'Primary Energy|Wind'])
    .aggregate(variable='Primary Energy')
)

df.append(
    solar_wind / primary_energy,
    variable='Primary Energy|Solar-Wind share', unit='-',
    inplace=True
)

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

def change_year(series, year):
    # When calculating changes over time, the year index should match
    # to be able to subtract two series of different years
    new_series = series.reset_index()
    new_series['year'] = year
    return new_series.set_index(pyam.utils.YEAR_IDX)['value']

def calc_increase_percentage(variable, year1, year2, suffix='|{}-{} change'):
    variable_year1 = to_series(df.filter(variable=variable, year=year1))
    variable_year2 = to_series(df.filter(variable=variable, year=year2))
    change_year1_year2 = (
        change_year(variable_year2, year1) - variable_year1
    ) / variable_year1
    df.append(
        change_year1_year2,
        variable=variable+suffix.format(year1, year2), unit='% change',
        inplace=True
    )


# Renewable share increase
calc_increase_percentage('Primary Energy|Renewables share', 2020, 2030)

# Solar-Wind share increase
calc_increase_percentage('Primary Energy|Solar-Wind share', 2020, 2030)

# Solar-Wind share increase
calc_increase_percentage('Secondary Energy|Electricity|Solar-Wind', 2020, 2030)

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
eip = to_series(
    df
    .filter(variable=['Emissions|CO2|Energy', 'Emissions|CO2|Industrial Processes'], scenario='CEDS')
    .aggregate(variable='Emissions|CO2')
)
df.append(
    eip,
    variable='Emissions|CO2|Energy and Industrial Processes', unit='Mt CO2/yr',
    inplace=True
)

# check presence of EIP in other scenarios
missing = df.require_variable(variable='Emissions|CO2|Energy and Industrial Processes')
missing = missing.loc[missing.model!='Reference',:]

# if missing, aggregate energy and ip
mapping = {}
for model in missing.model.unique():
    mapping[model] = list(missing.loc[missing.model==model, 'scenario'])

# Aggregate and add to the df
for model, scenarios in mapping.items():
    try:
        neweip = to_series(
            df.filter(model=model, scenario=scenarios,
                      variable=['Emissions|CO2|Energy', 'Emissions|CO2|Industrial Processes'],)
            .aggregate(variable='Emissions|CO2')
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

# # First, the aggregation tests ################################
if aggregation_variables is not None:
    for agg_name, agg_info in aggregation_variables.items():
        filter_check_aggregate(agg_info['variable'], agg_info['threshold'], agg_name,
           agg_info['key_historical'], agg_info['key_future'],
       )
else:
    print('Skipping aggregations')


# =============================================================================
# Add data: % increases
# =============================================================================

# Emissions|CO2 increase
calc_increase_percentage('Emissions|CO2', 2010, 2020)
# calc_increase_percentage('Emissions|CO2', 2015, 2020)
calc_increase_percentage('Emissions|CO2|Energy and Industrial Processes', 2010, 2020)

# Calculate CCS from energy (not industrial):
df.append(
    to_series(
        df
        .filter(variable=['Carbon Sequestration|CCS|Biomass', 'Carbon Sequestration|CCS|Fossil'])
        .aggregate(variable='Carbon Sequestration|CCS')
    ),
    variable='Carbon Sequestration|CCS|Biomass and Fossil', unit='Mt CO2/yr',
    inplace=True
)

    
#%%  =============================================================================
# Add data: Upper-Lower CEDS&EDGAR bounds (not used due to method)
# DO NOT DELETE, CAN BE USED TO CALCULATE THE REFERENCE VALUES WHICH ARE READ IN THE YAML FILE
# =============================================================================
# #Exclude wrongly extrapolated data and set empty values
# df = df.filter(model='Reference', scenario='CEDS', year=np.arange(2021,2101), keep=False)


# varis = ['Emissions|CO2', 'Emissions|CO2|Energy and Industrial Processes',
#           'Emissions|CH4']
# unit = 'Mt CO2/yr'


# for var in varis:
#     print(var)
#     ceds = df.filter(scenario='CEDS', variable=var, year=np.arange(2009,2021))
#     ceds.data.loc[ceds.data.year>2015, 'value'] = np.nan

#     ceds.data = ceds.data.interpolate(method ='spline', order=1, limit_direction = "both")
#     df = df.filter(model='Reference', scenario='CEDS', variable=var,
#                     year=np.arange(2016,2021), keep=False)
#     df.append(ceds.filter(year=np.arange(2016,2021)), inplace=True)

# yrs = [2015, 2018]

# for yr, var in it.product(yrs, varis):
#     dff = df.filter(model='Reference', scenario=['CEDS','EDGAR AR6'], year=yr, variable=var)
#     upper = np.max(dff.data['value'])
#     lower = np.min(dff.data['value'])
#     data = [['Reference','CEDS_EDGAR', 'World', var+'|Upper', unit, yr, upper],
#             ['Reference','CEDS_EDGAR', 'World', var+'|Lower', unit, yr, lower]]

#     data = pd.DataFrame(data=data, columns=pyam.IAMC_IDX+['year','value'])
#     data = pyam.IamDataFrame(data)
#     df.append(data, inplace=True)

# ab = df.filter(model='Reference',scenario='CEDS_EDGAR')
# ab.to_csv(f'{output_folder}ceds_edgar_extrapdata.csv')

#%% Second, for the reference checking
for agg_name, agg_info in reference_variables.items():
    df_ref = create_reference_df(
        agg_info['model'], agg_info['scenario'], agg_info['variables_threshold'].keys(), agg_info['ref_year']
    )
    filter_with_reference(
        df_ref, agg_info['variables_threshold'], agg_info['compare_year'], agg_name,
        agg_info['key_historical'], agg_info['key_future'],
    )


# Third, the bounds tests
for name, info in bounds_variables.items():
    filter_validate(info['variable'], info['year'], info['lo'], info['up'], name,
        info['key_historical'], info['key_future'], info['bound_threshold'],
   )


#%% Write out excel table

###################################
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


#%%
###################################
#
# Save to Excel and format output file
#
###################################


def strip_version(model):
    split = model.split(' ')
    if len(split) > 1:
        return ' '.join(split[:-1])
    return model
df.meta['model stripped'] = df.meta.reset_index()['model'].apply(strip_version).values

if modelbymodel==True:
    models = ['all'] + list(df.meta['model stripped'].unique())
else:
    models = ['all']

for model in models:
        modelstr = model.replace('/','-')
        xs = '' if model=='all' else 'teams\\'
        wbstr = f'{output_folder}{xs}vetting_flags_{modelstr}_{ver}.xlsx' 
        writer = pd.ExcelWriter(wbstr, engine='xlsxwriter')
        dfo = df.meta.reset_index().drop(['exclude','Source'], axis=1)
        if model != 'all':
            dfo = dfo.loc[dfo['model stripped'].isin([model, 'Reference'])]

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

plot_columns = {key: value for key, value in value_columns.items() if value['plot']}

fig = make_subplots(rows=len(plot_columns), cols=1)
color_column = 'model stripped'
nbins = 75
# fac = 0.5
dfo = df.meta.reset_index().drop('exclude', axis=1).copy()
try:
    colormap = dict(zip(dfo[color_column].unique(), px.colors.qualitative.D3))
except:
    colormap = None

in_legend = []
threshold_plot = 0.5

for i, (column, info) in enumerate(plot_columns.items()):
    

    xmin, xmax = None, None
    if info['ref_lo'] is not None and info['ref_up'] is not None:
        dx = info['ref_up'] - info['ref_lo']
        xmin = info['ref_lo'] - threshold_plot * dx
        xmax = info['ref_up'] + threshold_plot * dx
    elif info['ref_up'] is not None:
        xmax = info['ref_up'] * (1+threshold_plot)
    elif info['ref_lo'] is not None and info['unit'] in ['%', '% change']:
        xmax = 25 # Purely for visual purposes

    too_small = dfo[column] <= xmin
    too_large = dfo[column] >= xmax
    num_too_small = sum(too_small)
    num_too_large = sum(too_large)

    if num_too_small > 0:
        dfo.loc[too_small, column] = xmin
        fig.add_annotation(
            x=xmin, y=num_too_small,
            text="≤ x<sub>min</sub>",
            showarrow=True, arrowhead=7, ax=20, ay=-20,
            col=1, row=i+1
        )

    if num_too_large > 0:
        dfo.loc[too_large, column] = xmax
        fig.add_annotation(
            x=xmax, y=num_too_large,
            text="≥ x<sub>max</sub>",
            showarrow=True, arrowhead=7, ax=-20, ay=-20,
            col=1, row=i+1
        )


    # Create a histogram figure, and add all its traces to the main figure
    # Set bingroup to None to avoid sharing the same bins accross subplots
    for trace in px.histogram(dfo, x=column, color=color_column, nbins=nbins, color_discrete_map=colormap).data:
        fig.add_trace(trace.update(
            bingroup=None, showlegend=trace.showlegend & (i==0)
        ), col=1, row=i+1)


    # fig.update_xaxes(
    #     row=i+1, range=[xlo, xup])
        
    # Give xaxis its title and change tickformat if necessary
    fig.update_xaxes(
        row=i+1,
        title=column, title_standoff=0,
        tickformat='%' if info['unit'] in ['%', '% change'] else None
    )

    # Add reference, lower and upper bounds as lines, if they are defined:
    for name, color, label in zip(
        ['ref_value', 'ref_lo', 'ref_up'],
        ['black', 'mediumseagreen', 'mediumvioletred'],
        ['Reference', 'Lower bound', 'Upper bound']
    ):
        if info[name] is not None:
            domain = fig.layout['yaxis{}'.format(i+1)].domain
            fig.add_shape(
                type='line',
                xref='x{}'.format(i+1), yref='paper', # yref: stretch full height of plot, independent of y-range
                x0=info[name], x1=info[name], y0=domain[0], y1=domain[1],
                line_color=color
            )
        # Add legend items, not added when using shapes
        if i == 0:
            fig.add_scatter(x=[None, None], y=[None, None], name=label,mode='lines', line_color=color)

fig.update_yaxes(title='# scenarios', title_standoff=0)
fig.update_layout(
    height=200*len(plot_columns), width=900,
    barmode='stack',
    margin={'l': 50, 'r': 20, 't': 40, 'b': 60},
    legend_y=0.5
)

##%% Save to html
fig.update_layout(width=None).write_html(f'{output_folder}vetting_histograms_'+ver+'.html', include_plotlyjs='cdn')
print(time.time()-start)

df.to_excel(f'{output_folder}df_out.xlsx')

#%% Save emissions for climate assessment if infillerdb

if ver == 'infiller':
    fname = f'{output_folder}Emissions_for_climate_assessment_InfillerDB_input'
    
    dfca = dfin.filter(variable='Emissions*')
    dfca.filter(model=['Reference','MAGICC6'], keep=False, inplace=True)
    dfca.filter(year=range(2005,2101), inplace=True)
    dfca.load_meta(f'{output_folder}vetting_flags_all_infillerv6.xlsx', sheet_name='vetting_flags')
    # dfca.meta = dfca.meta[['exclude','vetting_infiller_v6']]
    dfca.filter(vetting_infiller_v6='PASS', keep=True, inplace=True)
    dfca = dfca.timeseries()
    to_add = [x for x in range(2005,2101) if x not in dfca.columns]
    for x in to_add:
        dfca[x] = np.nan
    colorder = list(range(2005,2101))
    dfca = dfca[colorder]
    # dfca = pyam.IamDataFrame(dfca)
    dfca.to_excel(fname+'.xlsx', merge_cells=False)

