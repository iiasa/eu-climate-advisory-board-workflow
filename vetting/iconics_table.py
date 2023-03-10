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

files = glob.glob(f'{main_folder}from_FabioS\\*EU_*.csv')
if len(files) == 1:
    dfin = pyam.IamDataFrame(files[0])
else:
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
# Filter years in case of odd zeros
years = range(2000,2101,5)
df.filter(year=years, inplace=True)

df.filter(model='GCAM*', scenario='*CurPol*', keep=False, inplace=True)

# vetting & climate
df.filter(OVERALL_binary='PASS', inplace=True)
df.meta.loc[df.meta['OVERALL_Assessment']=='Regional only', 'Category'] = 'regional'
df.filter(Category=['C1*','regional'], inplace=True)

# EUAB target
df.validate(criteria={'Emissions|Kyoto Gases (incl. indirect AFOLU)': {
    'up': 2061,
    'year': 2030}},
    exclude_on_fail=True)
df.validate(criteria={'Emissions|Kyoto Gases (incl. indirect AFOLU)': {
    'up': 300,
    'year': 2050}},
    exclude_on_fail=True)

df.filter(exclude=True, keep=False, inplace=True)

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

# Trade
components = [ 'Trade|Primary Energy|Biomass|Volume',
 'Trade|Primary Energy|Coal|Volume',
 'Trade|Primary Energy|Fossil',
 'Trade|Primary Energy|Gas|Volume',
 'Trade|Primary Energy|Oil|Volume',
 'Trade|Secondary Energy|Hydrogen|Volume',
 'Trade|Secondary Energy|Liquids|Biomass|Volume']
df.aggregate(variable='Trade', components=components)


## Fossil fuels

components = ['Primary Energy|Coal', 'Primary Energy|Gas', 'Primary Energy|Oil']
aggregate_missing_only(df, 'Primary Energy|Fossil', components, append=True)

components = ['Primary Energy|Coal|w/o CCS','Primary Energy|Oil|w/o CCS','Primary Energy|Gas|w/o CCS']
aggregate_missing_only(df, 'Primary Energy|Fossil|w/o CCS', components, append=True)



# primary Energy renewables
components = ['Primary Energy|Biomass', 'Primary Energy|Geothermal',
              'Primary Energy|Hydro', 'Primary Energy|Solar',
              'Primary Energy|Wind']

name = 'Primary Energy|Renewables (incl.Biomass)'
df.aggregate(name, components, append=True)
name = 'Primary Energy|Non-biomass renewables'
df.aggregate(name, components[1:], append=True)


# Final energy  fossil sectoral

components = ['Final Energy|Industry|Gases|Coal',
              'Final Energy|Industry|Gases|Natural Gas',
              'Final Energy|Industry|Liquids|Coal',
              'Final Energy|Industry|Liquids|Gas',
              'Final Energy|Industry|Liquids|Oil',
              'Final Energy|Industry|Solids|Coal',
              ]
name = 'Final Energy|Industry|Fossil'
df.aggregate(name, components=components, append=True)

# Res & Comm
components = ['Final Energy|Residential and Commercial|Gases|Coal',
              'Final Energy|Residential and Commercial|Gases|Natural Gas',
              'Final Energy|Residential and Commercial|Liquids|Coal',
              'Final Energy|Residential and Commercial|Liquids|Gas',
              'Final Energy|Residential and Commercial|Liquids|Oil',
              'Final Energy|Residential and Commercial|Solids|Coal',
              ]
name = 'Final Energy|Residential and Commercial|Fossil'
df.aggregate(name, components=components, append=True)

components = ['Final Energy|Transportation|Gases|Coal',
              'Final Energy|Transportation|Gases|Natural Gas',
              'Final Energy|Transportation|Liquids|Coal',
              'Final Energy|Transportation|Liquids|Gas',
              'Final Energy|Transportation|Liquids|Oil',
              ]
name = 'Final Energy|Transportation|Fossil'
df.aggregate(name, components=components, append=True)


# Trade
name = 'Trade'
components = [
             'Trade|Primary Energy|Biomass|Volume',
             'Trade|Primary Energy|Coal|Volume',
             'Trade|Primary Energy|Gas|Volume',
             'Trade|Primary Energy|Oil|Volume',
             'Trade|Secondary Energy|Hydrogen|Volume',
             'Trade|Secondary Energy|Liquids|Biomass|Volume']
df.aggregate(name, components=components, append=True)


name = 'Trade|Primary Energy|Fossil'
components = [
            'Trade|Primary Energy|Coal|Volume',
            'Trade|Primary Energy|Gas|Volume',
            'Trade|Primary Energy|Oil|Volume',]
df.aggregate(name, components=components, append=True)

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

# =============================================================================
#%% Cumulatuive emissions / sequestrations to 2050 values
# =============================================================================

df.interpolate(time=range(2000,2101), inplace=True)


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


# =============================================================================
#%% Non-CO2 % reduction 2020-2050
# =============================================================================
base_year = 2020
last_years = [2030, 2050]
for last_year in last_years:
    name = f'Non-CO2 emissions reductions {base_year}-{last_year} %'
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
            'Emissions|CO2|AFOLU',
             'Emissions|Kyoto Gases (incl. indirect AFOLU)',
             'Emissions|Kyoto Gases',
             'Carbon Sequestration|CCS',
             'Carbon Sequestration|CCS|Biomass',
             'Carbon Sequestration|CCS|Fossil',
             'Carbon Sequestration|CCS|Industrial Processes',

             ]
for x in indis_add:
    ynz_variables.append(x)


# =============================================================================
# Primary energy
# =============================================================================

ynz_variables.append('Primary Energy|Biomass')

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
#Hydrogen production as share of FE 
name = 'Hydrogen production|Final Energy|Share'
df.divide('Secondary Energy|Hydrogen', 'Final Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)

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

# =============================================================================
# Sectoral final energy fossil shares
# =============================================================================

# =============================================================================
# Industry
nv = 'Final Energy|Industry|Fossil|Share'
nu = '-'
df.divide('Final Energy|Industry|Fossil', 'Final Energy|Industry', 
          nv, 
          ignore_units=nu, 
          append=True)

ynz_variables.append(nv)

# =============================================================================
# Residential and Commercial
nv = 'Final Energy|Residential and Commercial|Fossil|Share'
nu = '-'
df.divide('Final Energy|Residential and Commercial|Fossil', 'Final Energy|Residential and Commercial', 
          nv, 
          ignore_units=nu, 
          append=True)

ynz_variables.append(nv)

# =============================================================================
# Transportation
nv = 'Final Energy|Transportation|Fossil|Share'
nu = '-'
df.divide('Final Energy|Transportation|Fossil', 'Final Energy|Transportation', 
          nv, 
          ignore_units=nu, 
          append=True)

ynz_variables.append(nv)

# =============================================================================
# Trade / imports
# =============================================================================
# =============================================================================
# Fossil fuel import dependency
nv = 'Primary Energy|Trade|Share'
df.divide('Trade', 'Primary Energy',
          nv,
          ignore_units='-',
          append=True)

df.multiply(nv, -100, 'PE Import dependency', ignore_units='%', append=True)
ynz_variables.append('PE Import dependency')



nv = 'Primary Energy|Fossil|Trade|Share'
df.divide('Trade|Primary Energy|Fossil', 'Primary Energy',
          nv,
          ignore_units='-',
          append=True)
df.multiply(nv, -100, 'PE Import dependency|Fossil', ignore_units='%', append=True)
ynz_variables.append('PE Import dependency|Fossil')


nv = 'Trade|Fossil|Share'
df.divide('Trade|Primary Energy|Fossil', 'Trade',
          nv,
          ignore_units='-',
          append=True)
ynz_variables.append(nv)

ynz_variables.append('Trade')
ynz_variables.append('Trade|Fossil|Share')
ynz_variables.append('Trade|Primary Energy|Fossil')



df.convert_unit('-', '%', factor=100, inplace=True)

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
    if v=='Emissions|Kyoto Gases (incl. indirect AFOLU)':
        name = f'{v} in 2030, {nu}'
        df.set_meta_from_data(name, variable=v, year=2030)
        
    name = f'{v} in 2050, {nu}'
    df.set_meta_from_data(name, variable=v, year=2050)
    


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

