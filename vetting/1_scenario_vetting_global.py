# -*- coding: utf-8 -*-
"""
Global Vetting script for Climate Advisory Board

Execute this script from within the "vetting" folder

"""
import os
# os.chdir('C:\\Github\\eu-climate-advisory-board-workflow\\vetting')
# Execute this script from within the "vetting" folder

#%% Import packages and data

import time
start = time.time()
print(start)
import numpy as np
import pyam
import pandas as pd
import yaml
# import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


log = True
modelbymodel = False    # Output single files for each model into Teams folder
single_model = False   # Testing mode- not used
include_data = True
print_log = print if log else lambda x: None
include_meta = False

from vetting_functions import *


#%% Configuration
# =============================================================================

#%% Settings for the specific run
region_level = 'global'
datestr = '20230712'

years = np.arange(2000, 2041, dtype=int).tolist()
year_aggregate = 2020

flag_fail = 'Fail'
flag_pass = 'Pass'
flag_pass_missing = 'Pass_missing'
flag_fail_missing = 'Fail_missing'

# the configuration file
config_vetting = f'{region_level}\\config_vetting_{region_level}.yaml'
instance = 'eu-climate-advisory-board'


input_data_ref = f'input_data\\input_reference_all.csv'

output_folder = f'outputs\\'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("created folder : ", output_folder)


if not os.path.exists(f'{output_folder}teams'):
    os.makedirs(f'{output_folder}teams')
    print("created folder : ", f'{output_folder}teams')

#%% Load data
#%% Settings for the project / dataset
# Specify what data to read in

varlist = ['Emissions|CO2',
            'Emissions|CO2|Energy and Industrial Processes',
            'Emissions|CO2|Energy',
            'Emissions|CO2|Industrial Processes',
            'Emissions|CH4',
            'Emissions|N2O',
            'Primary Energy',
            'Primary Energy|Fossil',
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

region = ['World']

#%%=============================================================================
# #%% Restart from here if necessary
# =============================================================================

#% load pyam data
dfin = pyam.read_iiasa(instance,
                        # model=models,
                        # scenario=scenarios,
                        variable=varlist,
                       year=years,
                       region=region, meta=False)


print('loaded from DB')
print(time.time()-start)

#%%
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
# Add additional reference data
# =============================================================================
dfref = pyam.IamDataFrame(input_data_ref).filter(region=region)
df.append(dfref, inplace=True)


# =============================================================================
## Secondary energy, electricity
# =============================================================================
secondary_energy_electricity = to_series(df.filter(variable='Secondary Energy|Electricity'))
secondary_wind_solar = to_series(
    df
    .filter(variable=['Secondary Energy|Electricity|Wind', 'Secondary Energy|Electricity|Solar'])
    .aggregate(variable='Secondary Energy|Electricity').filter(model='Reference', keep=False)
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

# =============================================================================
# Emissions & CCS
# =============================================================================

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
    

# # First, the aggregation tests ################################
if aggregation_variables is not None:
    for agg_name, agg_info in aggregation_variables.items():
        filter_check_aggregate(df.filter(year=year_aggregate),
                               agg_info['variable'],
                               agg_info['threshold'],
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
# Add data: % increases
# =============================================================================

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

#%%=============================================================================

#  #% Do checks against reference data
for agg_name, agg_info in reference_variables.items():
    df_ref = create_reference_df(df,
        agg_info['model'], agg_info['scenario'], agg_info['variables_threshold'].keys(), agg_info['ref_year'],                                                                                      
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
                          ver=region_level
        )
    # =============================================================================
    # Do the bounds checks
    # =============================================================================
#%%
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



#%% Write out excel table

###################################
#
# Create summary columns for historical and future
#
###################################

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

#% Overall - choose that only HISTORICAL == PASS
col = f'vetting_{region_level}'


df.meta.loc[(df.meta[meta_name_historical]==flag_pass) , col] = 'PASS'
df.meta.loc[(df.meta[meta_name_historical]==flag_pass_missing) , col] = flag_pass_missing
df.meta.loc[(df.meta[meta_name_historical]==flag_fail_missing) , col] = flag_fail_missing
df.meta.loc[(df.meta[meta_name_historical]==flag_fail) , col] = 'FAIL'


#%%
###################################
#
# Save to Excel and format output file
#
###################################

df.meta['model_stripped'] = df.meta.reset_index()['model'].apply(strip_version).values

if modelbymodel==True:
    models = ['all'] + list(df.meta['model_stripped'].unique())
else:
    models = ['all']

for model in models:
        modelstr = model.replace('/','-')
        xs = '' if model=='all' else 'teams\\'
        wbstr = f'{output_folder}{xs}vetting_flags_{modelstr}_{region_level}_{datestr}.xlsx'
        writer = pd.ExcelWriter(wbstr, engine='xlsxwriter')
        
        dfo = df.meta.reset_index().drop(['exclude'], axis=1) #,'Source' removed
        covcols = [x  for x in dfo.columns if 'coverage' in x]
        for x in ['exclude','source','Source','version','doi','reference']+covcols:
                dfo.drop(x, axis=1, errors='ignore', inplace=True)
        
        
        if model != 'all':
            dfo = dfo.loc[dfo['model_stripped'].isin([model, 'Reference'])]
# =============================================================================
#     # Write vetting flags sheet
# =============================================================================

        vetting_cols = ['model', 'model_stripped', 'scenario'] \
            + [ col, meta_name_historical, meta_name_future, flag_pass_missing, 
               flag_fail_missing] \
            + historical_columns \
            + future_columns 

        # dfo[['model', 'scenario']+[col, meta_name_historical, meta_name_future]+historical_columns+future_columns]
        dfo[vetting_cols].to_excel(writer, sheet_name='vetting_flags', index=False, header=True)
    # =============================================================================
    #     # Write details sheet
    # =============================================================================
        # dddddddd
        cols1 = dfo.columns.tolist()
        detail_cols = vetting_cols[:8]  + cols1[3:-2]

        dfo[detail_cols].to_excel(writer, sheet_name='details', index=False, startrow=1, header=False)
        md = pd.DataFrame(index=meta_docs.keys(), data=meta_docs.values(), columns=['Description'])
        md.to_excel(writer, sheet_name='description')

        if region_level != 'teams':
            worksheet = writer.sheets['vetting_flags']
            worksheet.set_column(0, 0, 13, None)
            worksheet.set_column(1, 1, 25, None)
            worksheet.set_column(2, len(dfo.columns)-1, 20, None)

    # =============================================================================
    #       # Model summary / detail pivot tables
    # =============================================================================
        if model == 'all':
            cols = ['model',  col, 'Key_historical','Key_future','IEA Primary Energy summary','IEA Electricity summary','EDGAR AR6 summary'] #'scenario',
            dfom = dfo.copy(deep=True)[cols]
            dfom = dfom.loc[dfom.model!='Reference',:]
            dfom.fillna('missing', inplace=True)
            
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
        
        cols = historical_columns + future_columns + dfop.columns[-4:-1].tolist() #+ dfop.columns[3:5].tolist()

        dfop_simple = dfop[cols].apply(pd.Series.value_counts).fillna(0).sort_index()
        dfop_simple = dfop_simple.T#[colorder]
        dfop_simple.rename(index={col: 'OVERALL'}, inplace=True)


        if region_level != 'teams':
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


        dfop_simple = dfop_simple.reindex(columns= cols)
        dfop_simple['sum_check'] = dfop_simple.sum(axis=1)
        dfop_simple.to_excel(writer, sheet_name='summary_pivot')

        # Add data
        if include_data:
            df.to_excel(writer, sheet_name='data', include_meta=include_meta)

# =============================================================================
#         ## Format the detail page
# =============================================================================

        # vetting_flags page
        worksheet = writer.sheets['vetting_flags']
        worksheet.set_column(0, 0, 13, None)
        worksheet.set_column(1, 1, 25, None)
        worksheet.set_column(2, len(dfo.columns)-1, 20, None)
        worksheet.freeze_panes(1, 3)
        worksheet.autofilter(0, 0, len(dfo), len(dfo.columns)-1)


        # Details
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
        for i, column in enumerate(detail_cols):
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

        if model=='all':
           os.startfile(wbstr)


###################################
#
#%% Create histograms
#
###################################

plot_columns = {key: value for key, value in value_columns.items() if value['plot']}

fig = make_subplots(rows=len(plot_columns), cols=1)
color_column = 'model_stripped'
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
fig.update_layout(width=None).write_html(f'{output_folder}vetting_histograms_{region_level}.html', include_plotlyjs='cdn')
print(time.time()-start)

df.to_excel(f'{output_folder}df_out_{region_level}.xlsx')

#%% Save emissions for climate assessment if infillerdb

if region_level == 'infiller':
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

