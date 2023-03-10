# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:32:48 2023

@author: byers
"""

# Summary table of final scenarios

import glob
import math
import io
import yaml
import re
import os
import pandas as pd
import numpy as np
import pyam
import plotly.express as px
import string
import itertools
import os
os.chdir('C:\\Github\\eu-climate-advisory-board-workflow\\vetting')
from vetting_functions import *


user = 'byers'

main_folder = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\'
output_folder = f'{main_folder}vetting\\'

wbstr = f'{output_folder}vetting_flags_global_regional_combined.xlsx'


#%% Load data
vetting = pd.read_excel(wbstr, sheet_name='Vetting_flags')

files = glob.glob(f'{main_folder}from_FabioS\\*climate_*.csv')

ct=0
for f in files:
    if ct==0:
        dfin = pyam.IamDataFrame(f)
    else:
        dfin = dfin.append(pyam.IamDataFrame(f))
    ct=ct+1
        
dfin.load_meta(wbstr, sheet_name='Vetting_flags')   

# =============================================================================
#%% Filter scenarios (RESTART FROM HERE)
# =============================================================================



df = dfin.filter(region='EU27')
years = range(2000,2101,5)
df.filter(year=years, inplace=True)


df.filter(OVERALL_binary='PASS', inplace=True)
df.meta.loc[df.meta['OVERALL_Assessment']=='Regional only', 'Category'] = 'regional'
df.filter(Category=['C1*','regional'], inplace=True)

meta_docs = {}


# =============================================================================
#%% Variable Aggregations 
# =============================================================================

# 'Emissions|CO2'
a = 'Emissions|CO2|Industrial Processes'
b = df.filter(variable=a).convert_unit('Mt CO2-equiv/yr', 'Mt CO2/yr')
df.filter(variable=a, keep=False, inplace=True)
df.append(b, inplace=True)

co2comps =  ['Emissions|CO2|Energy',
             'Emissions|CO2|Industrial Processes',
             'Emissions|CO2|LULUCF Direct+Indirect',
             # 'Emissions|CO2|LULUCF Indirect',
             ]

df.aggregate(variable='Emissions|CO2',
             components=co2comps, append=True)

# 'Emissions|CO2|Energy and Industrial Processes
components = [ 'Emissions|CO2|Industrial Processes', 'Emissions|CO2|Energy',]
df.aggregate(variable='Emissions|CO2|Energy and Industrial Processes',
             components=components, append=True)

# 'Emissions|Kyoto Gases'
df.subtract('Emissions|Kyoto Gases (incl. indirect AFOLU)',
            'Emissions|CO2',
            'tva',
            ignore_units='Mt CO2-equiv/yr',
            append=True)
df.subtract('tva',
            'Emissions|CO2|LULUCF Indirect',
            'Emissions|Kyoto Gases',
            ignore_units='Mt CO2-equiv/yr',
            append=True)
df.filter(variable='tva', keep=False, inplace=True
          )
# Emissions|Non-CO2
df.subtract('Emissions|Kyoto Gases (incl. indirect AFOLU)',
            'Emissions|CO2',
            'Emissions|Non-CO2',
            ignore_units='Mt CO2-equiv/yr',
            append=True)


#=============================================================================
#%% Year of netzero 
# =============================================================================

specdic = {'CO2': {'variable': 'Emissions|CO2',
                       'unitin': 'Mt CO2/yr',
                       'unitout': 'Gt CO2/yr',
                       'factor': 0.001},
           'CO2 EIP':{'variable': 'Emissions|CO2|Energy and Industrial Processes',
                                  'unitin': 'Mt CO2/yr',
                                  'unitout': 'Gt CO2/yr',
                                  'factor': 0.001},
           'CO2 AFOLU':{'variable': 'Emissions|CO2|AFOLU',
                                  'unitin': 'Mt CO2/yr',
                                  'unitout': 'Gt CO2/yr',
                                  'factor': 0.001},           
           'GHGs full':{'variable': 'Emissions|Kyoto Gases (incl. indirect AFOLU)',
                                  'unitin': 'Mt CO2-equiv/yr',
                                  'unitout': 'Gt CO2-equiv/yr',
                                  'factor': 0.001},
           'GHGs':{'variable': 'Emissions|Kyoto Gases',
                                  'unitin': 'Mt CO2-equiv/yr',
                                  'unitout': 'Gt CO2-equiv/yr',
                                  'factor': 0.001},   
           }

threshold = 0
nameNZCO2 = 'year of net-zero CO2 emissions (threshold=0 Gt CO2/yr)'
for indi, config in specdic.items():
    unitout = config['unitout']
    name = f'year of net-zero {indi} emissions (threshold={threshold} {unitout})'
    tsdata = filter_and_convert(df, config['variable'], unitin=config['unitin'], unitout=config['unitout'], factor=config['factor'])
    df.set_meta(tsdata.apply(year_of_net_zero, years=tsdata.columns, threshold=threshold, axis=1), name)

# nameNZ0eip = 'year of netzero CO2 EIP emissions (threshold=0 Gt CO2/yr)'
# co2eip = filter_and_convert(df, 'Emissions|CO2|Energy and Industrial Processes', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# df.set_meta(co2eip.apply(year_of_net_zero, years=co2eip.columns, threshold=0, axis=1), nameNZ0eip)

# nameNZ0eip = 'year of netzero CO2 EIP emissions (threshold=0 Gt CO2/yr)'
# co2eip = filter_and_convert(df, 'Emissions|CO2|Energy and Industrial Processes', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# df.set_meta(co2eip.apply(year_of_net_zero, years=co2eip.columns, threshold=0, axis=1), nameNZ0eip)
# =============================================================================
#%% Cumulatuive emissions / sequestrations to 2050 values
# =============================================================================

df.interpolate(time=range(2000,2101), inplace=True)

# co2 = filter_and_convert(df, 'Emissions|CO2', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# co2eip = filter_and_convert(df, 'Emissions|CO2|Energy and Industrial Processes', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# co2afolu = filter_and_convert(df, 'Emissions|CO2|AFOLU', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# # CS = filter_and_convert(df, 'Carbon Sequestration', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# ccs = filter_and_convert(df, 'Carbon Sequestration|CCS', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# beccs = filter_and_convert(df, 'Carbon Sequestration|CCS|Biomass', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# # ccsFI = filter_and_convert(df, 'Carbon Sequestration|CCS|Biomass', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)

# # seq_lu = filter_and_convert(df, 'Carbon Sequestration|Land Use', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# # dac =  filter_and_convert(df, 'Carbon Sequestration|Direct Air Capture', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# # ew =  filter_and_convert(df, 'Carbon Sequestration|Enhanced Weathering', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.00)

# ghgfull = filter_and_convert(df, 'Emissions|Kyoto Gases (incl. indirect AFOLU)', unitin='Mt CO2-equiv/yr', unitout='Gt CO2-equiv/yr', factor=0.001)

# ghg = filter_and_convert(df, 'Emissions|Kyoto Gases', unitin='Mt CO2-equiv/yr', unitout='Gt CO2-equiv/yr', factor=0.001)

#%% Cumulative calcs


specdic = {'net CO2': {'variable': 'Emissions|CO2',
                       'unitin': 'Mt CO2/yr',
                       'unitout': 'Gt CO2/yr',
                       'factor': 0.001},
           'net CO2 EIP':{'variable': 'Emissions|CO2|Energy and Industrial Processes',
                                  'unitin': 'Mt CO2/yr',
                                  'unitout': 'Gt CO2/yr',
                                  'factor': 0.001},
           'net CO2 AFOLU':{'variable': 'Emissions|CO2|AFOLU',
                                  'unitin': 'Mt CO2/yr',
                                  'unitout': 'Gt CO2/yr',
                                  'factor': 0.001},   
           'Non-CO2':{'variable': 'Emissions|Total Non-CO2',
                                  'unitin': 'Mt CO2-equiv/yr',
                                  'unitout': 'Gt CO2-equiv/yr',
                                  'factor': 0.001},            
           'CCS':{'variable': 'Carbon Sequestration|CCS',
                                  'unitin': 'Mt CO2/yr',
                                  'unitout': 'Gt CO2/yr',
                                  'factor': 0.001},
           'BECCS':{'variable': 'Carbon Sequestration|CCS|Biomass',
                                  'unitin': 'Mt CO2/yr',
                                  'unitout': 'Gt CO2/yr',
                                  'factor': 0.001},
           'GHGs full':{'variable': 'Emissions|Kyoto Gases (incl. indirect AFOLU)',
                                  'unitin': 'Mt CO2-equiv/yr',
                                  'unitout': 'Gt CO2-equiv/yr',
                                  'factor': 0.001},
           'GHGs':{'variable': 'Emissions|Kyoto Gases',
                                  'unitin': 'Mt CO2-equiv/yr',
                                  'unitout': 'Gt CO2-equiv/yr',
                                  'factor': 0.001},   
           }
           

# to 2050
baseyear = 2020
lastyear = 2050

for indi, config in specdic.items():
    variable = config['variable']
    tsdata = filter_and_convert(df, variable, unitin=config['unitin'], unitout=config['unitout'], factor=config['factor'])
    cumulative_unit = config['unitout'].split('/yr')[0]
    
    # to year of net-zero CO2
    name = f'cumulative {indi} ({baseyear} to year of net zero CO2, {cumulative_unit})'
    df.set_meta(tsdata.apply(lambda x: pyam.cumulative(x, first_year=baseyear, last_year=get_from_meta_column(df, x,                                                                nameNZCO2)), raw=False, axis=1), name)
    meta_docs[name] = f'Cumulative {indi} from {baseyear} until year of net zero CO2 (including the last year, {cumulative_unit}) ({variable})'
    
    # to 2050
    label = f'cumulative {indi} ({baseyear}-{lastyear}, {cumulative_unit})'
    df.set_meta(tsdata.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), label)
    meta_docs[name] = f'Cumulative {indi} from {baseyear} until {lastyear} (including the last year, {cumulative_unit}) ({variable})'




# # CO2
# cum_co2_label = 'cumulative net CO2 ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
# df.set_meta(co2.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_co2_label)
# meta_docs[cum_co2_label] = 'Cumulative net CO2 emissions from {} until {} (including the last year, {}) (native model Emissions|CO2)'.format(baseyear, lastyear, cumulative_unit)    

# # CO2 EIP
# cum_co2_label = 'cumulative net CO2 EIP ({}-{}, {}) (Native)'.format(baseyear, lastyear, cumulative_unit)
# df.set_meta(co2eip.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_co2_label)
# meta_docs[cum_co2_label] = 'Cumulative net CO2 EIP emissions from {} until {} (including the last year, {}) (native model Emissions|CO2)'.format(baseyear, lastyear, cumulative_unit)  


# # # Carbon Sequestration
# # cum_CS_label = 'cumulative Carbon Sequestration ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
# # df.set_meta(CS.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_CS_label)
# # meta_docs[cum_CS_label] = 'Cumulative carbon sequestration from {} until {} (including the last year, {})'        .format(baseyear, lastyear, cumulative_unit)

# # CCS
# cum_ccs_label = 'cumulative CCS ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
# df.set_meta(ccs.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_ccs_label)
# meta_docs[cum_ccs_label] = 'Cumulative carbon capture and sequestration from {} until {} (including the last year, {})'        .format(baseyear, lastyear, cumulative_unit)
# # BECCS
# cum_beccs_label = 'cumulative BECCS ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
# df.set_meta(beccs.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_beccs_label)
# meta_docs[cum_beccs_label] = 'Cumulative carbon capture and sequestration from bioenergy from {} until {} (including the last year, {})'.format(
#     baseyear, lastyear, cumulative_unit)   
# # # LU
# # name = 'cumulative sequestration land-use ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
# # df.set_meta(seq_lu.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), name)
# # meta_docs[name] = 'Cumulative carbon sequestration from land use from {} until {} (including the last year, {})'.format(
# #     baseyear, lastyear, cumulative_unit)    
# # # DAC
# # name = f'cumulative sequestration Direct Air Capture ({baseyear}-{lastyear}, {cumulative_unit})'
# # df.set_meta(dac.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), name)
# # meta_docs[name] = 'Cumulative carbon sequestration from Direct Air Capture from {} until {} (including the last year, {})'.format(
# #     baseyear, lastyear, cumulative_unit)
# # # EW
# # name = f'cumulative sequestration Enhanced Weathering ({baseyear}-{lastyear}, {cumulative_unit})'
# # df.set_meta(ew.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), name)
# # meta_docs[name] = 'Cumulative carbon sequestration from Enhanced Weathering from {} until {} (including the last year, {})'.format(
# #     baseyear, lastyear, cumulative_unit)

# cum_ghgfull = 'cumulative Kyoto Gases (incl. indirect AFOLU) ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
# df.set_meta(cum_ghgfull.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_beccs_label)
# meta_docs[cum_beccs_label] = 'Cumulative Kyoto Gases (incl. indirect AFOLU) from {} until {} (including the last year, {})'.format(
#     baseyear, lastyear, cumulative_unit) 

# cum_ghg = 'cumulative Kyoto Gases ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
# df.set_meta(cum_ghgfull.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_beccs_label)
# meta_docs[cum_beccs_label] = 'Cumulative Kyoto Gases from {} until {} (including the last year, {})'.format(
#     baseyear, lastyear, cumulative_unit) # 

# =============================================================================
#%% cumulative CO2/GHGs to net zero CO2 
# =============================================================================

# name = 'cumulative net CO2 to year of net zero CO2, Gt CO2'
# df.set_meta(co2.apply(lambda x: pyam.cumulative(x, first_year=baseyear, last_year=get_from_meta_column(df, x,                                                                nameNZ0)), raw=False, axis=1), name)

# name = 'cumulative Kyoto Gases (incl. indirect AFOLU) to year of net zero CO2, Gt CO2-equiv'
# df.set_meta(ghgfull.apply(lambda x: pyam.cumulative(x, first_year=baseyear, last_year=get_from_meta_column(df, x,                                                                nameNZ0)), raw=False, axis=1), name)

# name = 'cumulative Kyoto Gases (incl. indirect AFOLU) to year of net zero CO2, Gt CO2-equiv'
# df.set_meta(ghg.apply(lambda x: pyam.cumulative(x, first_year=baseyear, last_year=get_from_meta_column(df, x,                                                                nameNZ0)), raw=False, axis=1), name)

# =============================================================================
#%% Non-CO2 % reduction 2020-2050
# =============================================================================
base_year = 2020
last_years = [2030, 2050]
for last_year in last_years:
    name = f'Non-CO2 emissions reductions {base_year}-{last_year} % '
    a = df.filter(variable='Emissions|Total Non-CO2').timeseries()
    rd = 100* (1-(a[last_year] / a[base_year]))
    df.set_meta(rd, name, )
    


# =============================================================================
#%% Calculation of indicator variables
# =============================================================================
ynz_variables = []

# =============================================================================
# Emissions
# =============================================================================
indis_add = [
            'Emissions|CO2',
            'Emissions|Total Non-CO2',
             'Emissions|Kyoto Gases (incl. indirect AFOLU)',
             'Emissions|Kyoto Gases',

             ]
for x in indis_add:
    ynz_variables.append(x)

# =============================================================================
# 'Emissions|CO2|AFOLU'  
ynz_variables.append('Emissions|CO2|AFOLU')

# =============================================================================
# 'Carbon Sequestration|CCS|Biomass',
ynz_variables.append('Carbon Sequestration|CCS|Biomass')

# =============================================================================
# Primary energy
# =============================================================================
missing = df.require_variable(variable='Primary Energy|Fossil')
if missing is not None:
    missing = missing.loc[missing.model!='Reference',:]
    components = ['Primary Energy|Coal', 'Primary Energy|Gas', 'Primary Energy|Oil']
    # if missing, aggregate energy and ip
    mapping = {}
    for model in missing.model.unique():
        mapping[model] = list(missing.loc[missing.model==model, 'scenario'])
    
    # Aggregate and add to the df
    if len(mapping)>0:
        for model, scenarios in mapping.items():
            try:
                newpef = to_series(
                    df.filter(model=model, scenario=scenarios, variable=components).aggregate(variable='Primary Energy|Fossil', components=components)
                    )
        
                df.append(
                    newpef,
                variable='Primary Energy|Fossil', unit='EJ/yr',
                inplace=True
                )
            except(IndexError):
                print('No components:{},{}'.format(model, scenarios))

components = ['Primary Energy|Biomass', 'Primary Energy|Geothermal',
              'Primary Energy|Hydro', 'Primary Energy|Solar',
              'Primary Energy|Wind']

name = 'Primary Energy|Renewables (incl.Biomass)'
df.aggregate(name, components, append=True)
name = 'Primary Energy|Non-biomass renewables'
df.aggregate(name, components[1:], append=True)


# =============================================================================
# Primary energy - Renewables share

name = 'Primary Energy|Renewables (incl.Biomass)|Share'
df.divide('Primary Energy|Renewables (incl.Biomass)', 'Primary Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)

name = 'Primary Energy|Non-biomass renewables|Share'
df.divide('Primary Energy|Non-biomass renewables', 'Primary Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)

# =============================================================================
# Primary energy - Fossil share

name = 'Primary Energy|Fossil|Share'
df.divide('Primary Energy|Fossil', 'Primary Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)


name = 'Primary Energy|Fossil|w/o CCS|Share'
df.divide('Primary Energy|Fossil|w/o CCS', 'Primary Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)




# =============================================================================
# Secondary energy electricity renewables & hydrogen
# =============================================================================


# Secondary energy Renewables
# Drop non-bio renewables
df.filter(variable='Secondary Energy|Electricity|Renewables (incl.Biomass)', 
              keep=False, inplace=True)
df.filter(variable='Secondary Energy|Electricity|Non-Biomass Renewables', 
              keep=False, inplace=True)

rencomps = [
      'Secondary Energy|Electricity|Biomass',
      'Secondary Energy|Electricity|Geothermal',
      'Secondary Energy|Electricity|Hydro',
      'Secondary Energy|Electricity|Solar',
      'Secondary Energy|Electricity|Wind',]

df.aggregate('Secondary Energy|Electricity|Renewables (incl.Biomass)', 
                components=rencomps, append=True)

df.aggregate('Secondary Energy|Electricity|Non-Biomass Renewables', 
                components=rencomps[1:], append=True)



# % of renewables in electricity
nv = 'Secondary Energy|Electricity|Renewables (incl.Biomass)|Share'
# nu = '%'
df.divide('Secondary Energy|Electricity|Renewables (incl.Biomass)', 
          'Secondary Energy|Electricity', 
          nv,
          ignore_units='-',
          append=True)
ynz_variables.append(nv)

# % of non-bio renewables in electricity
nv = 'Secondary Energy|Electricity|Non-Biomass Renewables|Share'
# nu = '%'
df.divide('Secondary Energy|Electricity|Non-Biomass Renewables', 
          'Secondary Energy|Electricity',
          nv,
          ignore_units='-',
          append=True)
ynz_variables.append(nv)

# =============================================================================
# Hydrogen production as share of FE 
# name = 'Hydrogen production|Final Energy|Share'
# df.divide('Secondary Energy|Hydrogen', 'Final Energy',
#           name, ignore_units='-',
#           append=True)
# ynz_variables.append(name)

# =============================================================================
# Final energy
# =============================================================================

# =============================================================================
# #final energy / capita

# nv='Final Energy|Per capita'
# nu='EJ/yr /person'
# print(nv)
# df.divide('Final Energy', 'Population', 
#           nv,
#           ignore_units=nu,
#           append=True)
# ynz_variables.append(nv)


# =============================================================================
# # #GDP / unit Final Energy

# nv = 'GDP|MER|Final Energy'
# nu = 'US$2010/MJ'
# df.divide('GDP|MER', 'Final Energy',
#           nv,
#           ignore_units=nu,
#           append=True)
# ynz_variables.append(nv)

ynz_variables.append('Final Energy')
# ynz_variables.append('Secondary Energy|Hydrogen')
# =============================================================================
# #% of final energy that is electrified

nv = 'Final Energy|Electrification|Share'
nu = '-'
df.divide('Final Energy|Electricity', 'Final Energy', 
          nv, 
          ignore_units=nu, 
          append=True)

ynz_variables.append(nv)

df.convert_unit('-', '%', factor=100, inplace=True)


# =============================================================================
# Trade / imports
# =============================================================================

# =============================================================================
# Fossil fuel imports
name = 'Trade|Primary Energy|Fossil'
components = [
'Trade|Primary Energy|Coal|Volume',
'Trade|Primary Energy|Gas|Volume',
'Trade|Primary Energy|Oil|Volume',]
df.aggregate(name, components=components)


# =============================================================================
#%% Calculate indicators in year of net-zero and 2050
# =============================================================================


df.interpolate(time=range(2000,2101), inplace=True)

for v in ynz_variables:

    datats = filter_and_convert(df, v)
    nu = datats.reset_index().unit.unique()[0]
    
    name = f'{v} in year of net zero, {nu}'
    df.set_meta(datats.apply(lambda x: x[get_from_meta_column(df, x,
                                                              nameNZCO2)],
                                            raw=False, axis=1), name)    
    name = f'{v} in 2050, {nu}'
    df.set_meta_from_data(name, variable=v, year=2050)
    

                                            
#%%
# =============================================================================
# #%% Multiply all share columns by 100% 
# # =============================================================================

# pccols = [x for x in list(df.meta.columns) if "Share" in x]
# df.meta[pccols] = df.meta[pccols]*100
# npccols = [x.replace('-','%') for x in pccols]
# rncols = {k:v for k,v in zip(pccols, npccols)}
# df.meta.rename(columns=rncols, inplace=True)




# =============================================================================
#%% Write out 
# =============================================================================
fn = f'{main_folder}iconics\\iconics_NZ_data_and_table.xlsx'
writer = pd.ExcelWriter(fn, engine='xlsxwriter')


df.to_excel(writer, sheet_name='data', include_meta=True)



# meta page
worksheet = writer.sheets['meta']
worksheet.set_column(0, 0, 20, None)
worksheet.set_column(1, 1, 25, None)
worksheet.freeze_panes(1, 2)
worksheet.autofilter(0, 0, len(df.meta), len(df.meta.columns)+1)

workbook = writer.book
header_format_creator = lambda is_bold: workbook.add_format({
    'bold': is_bold,
    'text_wrap': True,
    'align': 'center',
    'valign': 'top',
    'border': 1
})
header_format = header_format_creator(True)
# subheader_format = header_format_creator(False)

for col_num, value in enumerate(df.meta.columns):
    # curr_format = subheader_format if value[0] == '(' or value[-1] == ']' else header_format
    worksheet.write(0, col_num+2, value, header_format) 
worksheet.set_column(2, len(df.meta.columns)+1, 15, None)



letters = pd.Series(list(
    itertools.chain(
        string.ascii_uppercase, 
        (''.join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2))
)))


end = len(df.meta)+1

# Cumulative columns
end_col = len(df.meta.columns)+2
letters_cum = letters[18:54]

refs = [f'{c}2:{c}{end}' for c in letters_cum]
for ref in refs:
    worksheet.conditional_format(ref, {'type': '3_color_scale',
                                       'min_color':'#6292bf',
                                           'mid_color':'#FFFFFF',
                                           'max_color':'#fc6060'})
    
letters_indis = letters[54:end_col]
refs = [f'{c}2:{c}{end}' for c in letters_indis]
for ref in refs:
    worksheet.conditional_format(ref, {'type': '3_color_scale',
                                       'min_color':'#6292bf',
                                           'mid_color':'#FFFFFF',
                                           'max_color':'#56ba49'})

# Change format of value columns
integer_format = workbook.add_format({'num_format': '0'})
largenum_format = workbook.add_format({'num_format': '0'})
smallnum_format = workbook.add_format({'num_format': '0.0'})
percentage_format = workbook.add_format({'num_format': '0'})
# percentage_change_format = workbook.add_format({'num_format': '+0%;-0%;0%'})

start = 18
endcol = len(df.meta.columns)+1
value_columns = list(enumerate(df.meta.columns))[start:]
for i, column in value_columns:#enumerate(value_columns):
    # col = i+start
    i=i+2
    if '%' in column:
        worksheet.set_column(i, i, None, percentage_format)
    elif 'threshold' in column:
        worksheet.set_column(i, i, None, integer_format)
    elif df.meta[column].dtype == float and abs(df.meta[column].median()) > 10:
        worksheet.set_column(i, i, None, largenum_format)
    elif df.meta[column].dtype == float and abs(df.meta[column].median()) <= 10:
        worksheet.set_column(i, i, None, smallnum_format)







# data page
# if model!= 'all':
worksheet = writer.sheets['data']
worksheet.set_column(0, 1, 25, None)
# worksheet.set_column(1, 1, 20, None)
worksheet.set_column(2, 2, 8, None)
worksheet.set_column(3, 3, 30, None)
worksheet.set_column(4, 4, 12, None)
worksheet.set_column(5, -1, 8, None)
worksheet.freeze_panes(1, 2)
worksheet.autofilter(0, 0, len(df.meta), len(df.year))


writer.close()

os.startfile(fn)


df.meta.to_excel(f'{main_folder}iconics\\iconics_NZ_table.xlsx',
                 sheet_name='meta')

