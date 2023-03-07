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

#%% Filter scenarios

df = dfin.filter(region='EU27')

df.filter(OVERALL_binary='PASS', inplace=True)
df.filter(Category='C1*', inplace=True)

meta_docs = {}

# =============================================================================
#%% Region Aggregations  to EU
# =============================================================================
# for v in df.variable:
#     df.aggregate_region(variable=v,
#                              region='EU', 
#                              append=True)
                            

# =============================================================================
#%% Variable Aggregations 
# =============================================================================
a = 'Emissions|CO2|Industrial Processes'
b = df.filter(variable=a).convert_unit('Mt CO2-equiv/yr', 'Mt CO2/yr')
df.filter(variable=a, keep=False, inplace=True)
df.append(b, inplace=True)

co2comps =  ['Emissions|CO2|AFOLU',
             'Emissions|CO2|Energy',
             'Emissions|CO2|Industrial Processes',
             'Emissions|CO2|LULUCF Direct+Indirect',
             # 'Emissions|CO2|LULUCF Indirect',
             ]

df.aggregate(variable='Emissions|CO2',
             components=co2comps, append=True)


components = [ 'Emissions|CO2|Industrial Processes', 'Emissions|CO2|Energy',]
df.aggregate(variable='Emissions|CO2|Energy and Industrial Processes',
             components=components, append=True)






# =============================================================================
#%% Cumulatuive emissions / sequestrations to 2050 values
# =============================================================================

df.interpolate(time=range(2000,2101), inplace=True)

co2 = filter_and_convert(df, 'Emissions|CO2', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
co2eip = filter_and_convert(df, 'Emissions|CO2|Energy and Industrial Processes', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
co2afolu = filter_and_convert(df, 'Emissions|CO2|AFOLU', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# CS = filter_and_convert(df, 'Carbon Sequestration', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
ccs = filter_and_convert(df, 'Carbon Sequestration|CCS', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
beccs = filter_and_convert(df, 'Carbon Sequestration|CCS|Biomass', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# ccsFI = filter_and_convert(df, 'Carbon Sequestration|CCS|Biomass', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)

# seq_lu = filter_and_convert(df, 'Carbon Sequestration|Land Use', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# dac =  filter_and_convert(df, 'Carbon Sequestration|Direct Air Capture', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.001)
# ew =  filter_and_convert(df, 'Carbon Sequestration|Enhanced Weathering', unitin='Mt CO2/yr', unitout='Gt CO2/yr', factor=0.00)
# 

#%% Cumulative to 2050

cumulative_unit = 'Gt CO2'
baseyear = 2020
lastyear = 2050

# CO2
cum_co2_label = 'cumulative net CO2 ({}-{}, {}) (Native)'.format(baseyear, lastyear, cumulative_unit)
df.set_meta(co2.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_co2_label)
meta_docs[cum_co2_label] = 'Cumulative net CO2 emissions from {} until {} (including the last year, {}) (native model Emissions|CO2)'.format(baseyear, lastyear, cumulative_unit)    

# CO2 EIP
cum_co2_label = 'cumulative net CO2 EIP ({}-{}, {}) (Native)'.format(baseyear, lastyear, cumulative_unit)
df.set_meta(co2eip.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_co2_label)
meta_docs[cum_co2_label] = 'Cumulative net CO2 EIP emissions from {} until {} (including the last year, {}) (native model Emissions|CO2)'.format(baseyear, lastyear, cumulative_unit)  


# # Carbon Sequestration
# cum_CS_label = 'cumulative Carbon Sequestration ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
# df.set_meta(CS.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_CS_label)
# meta_docs[cum_CS_label] = 'Cumulative carbon sequestration from {} until {} (including the last year, {})'        .format(baseyear, lastyear, cumulative_unit)

# CCS
cum_ccs_label = 'cumulative CCS ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
df.set_meta(ccs.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_ccs_label)
meta_docs[cum_ccs_label] = 'Cumulative carbon capture and sequestration from {} until {} (including the last year, {})'        .format(baseyear, lastyear, cumulative_unit)
# BECCS
cum_beccs_label = 'cumulative BECCS ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
df.set_meta(beccs.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), cum_beccs_label)
meta_docs[cum_beccs_label] = 'Cumulative carbon capture and sequestration from bioenergy from {} until {} (including the last year, {})'.format(
    baseyear, lastyear, cumulative_unit)   
# # LU
# name = 'cumulative sequestration land-use ({}-{}, {})'.format(baseyear, lastyear, cumulative_unit)
# df.set_meta(seq_lu.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), name)
# meta_docs[name] = 'Cumulative carbon sequestration from land use from {} until {} (including the last year, {})'.format(
#     baseyear, lastyear, cumulative_unit)    
# # DAC
# name = f'cumulative sequestration Direct Air Capture ({baseyear}-{lastyear}, {cumulative_unit})'
# df.set_meta(dac.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), name)
# meta_docs[name] = 'Cumulative carbon sequestration from Direct Air Capture from {} until {} (including the last year, {})'.format(
#     baseyear, lastyear, cumulative_unit)
# # EW
# name = f'cumulative sequestration Enhanced Weathering ({baseyear}-{lastyear}, {cumulative_unit})'
# df.set_meta(ew.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), name)
# meta_docs[name] = 'Cumulative carbon sequestration from Enhanced Weathering from {} until {} (including the last year, {})'.format(
#     baseyear, lastyear, cumulative_unit)



# =============================================================================
#%% Year of netzero 
# =============================================================================

nameNZ0 = 'year of netzero CO2 emissions (threshold=0 Gt CO2/yr)'
df.set_meta(co2.apply(year_of_net_zero, years=co2.columns, threshold=0, axis=1), nameNZ0)

nameNZ0 = 'year of netzero CO2 EIP emissions (threshold=0 Gt CO2/yr)'
df.set_meta(co2eip.apply(year_of_net_zero, years=co2eip.columns, threshold=0, axis=1), nameNZ0)