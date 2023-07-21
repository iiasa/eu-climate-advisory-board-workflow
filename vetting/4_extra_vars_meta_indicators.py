# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:32:48 2023

@author: byers

# Execute this script from within the "vetting" folder

# The aim of this script is to pull original data from the db, and demonstrate the 
calculation of additional variables and metadata indicators that are present.

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
import itertools
import os
import matplotlib.pyplot as plt
import seaborn as sns
# os.chdir('C:\\Github\\eu-climate-advisory-board-workflow\\vetting')
from vetting_functions import *


output_folder = f'outputs\\'
instance = 'eu-climate-advisory-board'
datestr = '20230712'  
fn_out = f'{output_folder}iconics_NZ_data_and_table_{datestr}_v17_reprod_meta.xlsx'


#%% Load data

years = list(range(2015,2071))

prefix = 'Diagnostics|Harmonized|'

# Variables that we will pull from the database and use for calculations
varlist = ['Diagnostics*',
           'Final Energy*',
           'Trade|*',
           'Imports*',
           'Carbon Sequestration*',
           'Secondary Energy|Hydrogen',
           'Carbon Capture|Usage', 
           'Carbon Capture|Storage',
           ]


# New variables that will be created (will be used for comparison later)
new_vars = [
        'Carbon Capture, Use, and Sequestration', 
        'Primary Energy|*Share',
        'Secondary Energy|Electricity|*Share',
        'Final Energy|*Share',
        'Final Energy|Industry|Fossil',
        'Final Energy|Residential and Commercial|Fossil',
        'Final Energy|Transportation|Fossil',
        'Hydrogen production|Final Energy|Share',
        'Trade|Primary Energy|Fossil',
        'PE Import dependency*',
        'Primary Energy|Import Dependency*',
        'Trade',
        ]


#% load pyam data from database
dfin = pyam.read_iiasa(instance,
                        # model=models,
                        # scenario=scenarios,
                        variable=varlist,
                        year=years,
                        region='EU27', 
                        meta=True)


# For testing - make sure drop all data with "Share", which is always calculated
dfin.filter(variable='*Share', keep=False, inplace=True)

# Keep only the trade data in 'EJ/yr'
dfin.filter(variable='Trade*', unit='billion US$2010/yr', 
            keep=False, inplace=True)


#%% Filter scenarios (RESTART FROM HERE) 
# =============================================================================

df = dfin.filter(region='EU27')

# Filter years in case of odd zeros
years = list(range(1990,2016,5))+[2019]+list(range(2020,2071,5))
df.filter(year=years, inplace=True)


# Keep only scenarios that passed vetting = equivalent to 'OVERALL_binary'
# column that outputs from script 3_merge_summaries.py
df.filter(**{'Vetting status':'pass'}, inplace=True)

# Keep only those are are C1 climate, and/or regional models (GCAM-PR 5.3)
df.filter(Category=['C1*', 'Regional only'], inplace=True)

meta_docs = {}

# =============================================================================
#%% Variable Aggregations (skip this whole section?)
# # =============================================================================
    
# =============================================================================
# Carbon Capture, Use, and Sequestration
# # For REMIND 2.1 / 3,2 scenarios
# =============================================================================
# This is done due to different reporting by these REMIND scenarios
df.aggregate('Carbon Capture, Use, and Sequestration', 
             components=['Carbon Capture|Usage', 
                         'Carbon Capture|Storage'],
                          append=True)
    
# Other models - Get CCS and add DACCS
dfo = df.filter(model=['REMIND 2.1', 'REMIND 3.2'], keep=False)
dfo.aggregate('Carbon Capture, Use, and Sequestration',
              ['Carbon Sequestration|CCS', 
               'Carbon Sequestration|Direct Air Capture'],  
              append=True)
df.append(dfo.filter(variable='Carbon Capture, Use, and Sequestration'), 
          inplace=True)
    
    
# Primary Energy renewables
components = [f'{prefix}Primary Energy|Biomass', 
              f'{prefix}Primary Energy|Geothermal',
              f'{prefix}Primary Energy|Hydro', 
              f'{prefix}Primary Energy|Solar',
              f'{prefix}Primary Energy|Wind',
              ]
    
name = f'{prefix}Primary Energy|Renewables (incl.Biomass)'
df.aggregate(name, components, append=True)
    
    
# Final energy  fossil sectoral
components = ['Final Energy|Industry|Gases|Coal',
              'Final Energy|Industry|Gases|Natural Gas',
              'Final Energy|Industry|Liquids|Coal',
              'Final Energy|Industry|Liquids|Gas',
              'Final Energy|Industry|Liquids|Oil',
              'Final Energy|Industry|Solids|Coal',
              ]
name = 'Final Energy|Industry|Fossil'
aggregate_missing_only(df, name, components, append=True)
    
# Res & Comm
components = ['Final Energy|Residential and Commercial|Gases|Coal',
              'Final Energy|Residential and Commercial|Gases|Natural Gas',
              'Final Energy|Residential and Commercial|Liquids|Coal',
              'Final Energy|Residential and Commercial|Liquids|Gas',
              'Final Energy|Residential and Commercial|Liquids|Oil',
              'Final Energy|Residential and Commercial|Solids|Coal',
              ]
name = 'Final Energy|Residential and Commercial|Fossil'
aggregate_missing_only(df, name, components, append=True)
    
# Transportation
components = ['Final Energy|Transportation|Gases|Coal',
              'Final Energy|Transportation|Gases|Natural Gas',
              'Final Energy|Transportation|Liquids|Coal',
              'Final Energy|Transportation|Liquids|Gas',
              'Final Energy|Transportation|Liquids|Oil',
              ]
name = 'Final Energy|Transportation|Fossil'
aggregate_missing_only(df, name, components, append=True)
    
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
aggregate_missing_only(df, name, components, append=True)

#=============================================================================
#%% Year of netzero calculations
# =============================================================================

# Calculate year of net zeros
specdic = {
    'CO2': {
        'variable': f'{prefix}Emissions|CO2',
        'unitin': 'Mt CO2/yr',
        'unitout': 'Gt CO2/yr',
        'factor': 0.001
    },
    'GHGs* full': {
        'variable': f'{prefix}Emissions|Kyoto Gases (AR4) (EEA)',
        'unitin': 'Mt CO2-equiv/yr',
        'unitout': 'Gt CO2-equiv/yr',
        'factor': 0.001
    },
    'GHGs** full': {
        'variable': f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)',
        'unitin': 'Mt CO2-equiv/yr',
        'unitout': 'Gt CO2-equiv/yr',
        'factor': 0.001
    }
}


threshold = 0
nameNZCO2 = f'year of net-zero CO2 emissions (threshold={threshold} Gt CO2/yr)'

for key, value in specdic.items():
    variable = value['variable']
    unitin = value['unitin']
    unitout = value['unitout']
    factor = value['factor']
    
    tsdata = filter_and_convert(df, variable, unitin=unitin, unitout=unitout, factor=factor)
    name = f'year of net-zero {key} emissions (threshold={threshold} {unitout})'
    
    df.set_meta(tsdata.apply(year_of_net_zero, years=tsdata.columns, threshold=threshold, axis=1), name)


# =============================================================================
#%% Cumulatuive emissions / sequestrations to 2050 values
# =============================================================================

specdic = {
    'net CO2': {
        'variable': f'{prefix}Emissions|CO2',
        'unitin': 'Mt CO2/yr',
        'unitout': 'Gt CO2/yr',
        'factor': 0.001
    },
    'Non-CO2': {
        'variable': f'{prefix}Emissions|Total Non-CO2',
        'unitin': 'Mt CO2-equiv/yr',
        'unitout': 'Gt CO2-equiv/yr',
        'factor': 0.001
    },
    'CCS': {
        'variable': f'Carbon Sequestration|CCS',
        'unitin': 'Mt CO2/yr',
        'unitout': 'Gt CO2/yr',
        'factor': 0.001
    },
    'BECCS': {
        'variable': f'Carbon Sequestration|CCS|Biomass',
        'unitin': 'Mt CO2/yr',
        'unitout': 'Gt CO2/yr',
        'factor': 0.001
    },
    'GHGs* (incl. indirect AFOLU)': {
        'variable': f'{prefix}Emissions|Kyoto Gases (AR4) (EEA)',
        'unitin': 'Mt CO2-equiv/yr',
        'unitout': 'Gt CO2-equiv/yr',
        'factor': 0.001
    },
    'GHGs** (incl. indirect AFOLU)': {
        'variable': f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)',
        'unitin': 'Mt CO2-equiv/yr',
        'unitout': 'Gt CO2-equiv/yr',
        'factor': 0.001
    },
    'CO2 FFI': {
        'variable': f'{prefix}Emissions|CO2|Energy and Industrial Processes', 
        'unitin': 'Mt CO2-equiv/yr',
        'unitout': 'Gt CO2-equiv/yr',
        'factor': 0.001
    },
    'AFOLU (direct+indirect)': {
        'variable': f'{prefix}Emissions|CO2|LULUCF Direct+Indirect',
        'unitin': 'Mt CO2/yr',
        'unitout': 'Gt CO2/yr',
        'factor': 0.001
    }
}

baseyear = 2020
lastyear = 2050

for indi, config in specdic.items():
    variable = config['variable']
    unitin = config['unitin']
    unitout = config['unitout']
    factor = config['factor']
    
    tsdata = filter_and_convert(df, variable, unitin=unitin, unitout=unitout, factor=factor)
    cumulative_unit = unitout.split('/yr')[0]
    
    # to year of net-zero CO2
    label = f'cumulative {indi} ({baseyear} to year of net zero CO2, {cumulative_unit})'
    df.set_meta(
        tsdata.apply(
            lambda x: pyam.cumulative(x, first_year=baseyear, last_year=get_from_meta_column(df, x, nameNZCO2)),
            raw=False,
            axis=1
        ),
        label
    )
    meta_docs[label] = f'Cumulative {indi} from {baseyear} until year of net zero CO2 (including the last year, {cumulative_unit}) ({variable})'
    
    # to 2050
    label = f'cumulative {indi} ({baseyear}-{lastyear}, {cumulative_unit})'
    df.set_meta(
        tsdata.apply(
            pyam.cumulative,
            raw=False,
            axis=1,
            first_year=baseyear,
            last_year=lastyear
        ),
        label
    )
    meta_docs[label] = f'Cumulative {indi} from {baseyear} until {lastyear} (including the last year, {cumulative_unit}) ({variable})'

    # 2020 to 2030 and 2030 to 2050
    if indi in ['net CO2', 'GHGs (incl. indirect AFOLU)', 
                'GHGs* (incl. indirect AFOLU)', 
                'GHGs** (incl. indirect AFOLU)', 
                'AFOLU (direct+indirect)']:
        
        label1 = f'cumulative {indi} (2020-2030, {cumulative_unit})'
        label2 = f'cumulative {indi} (2030-{lastyear}, {cumulative_unit})'
        
        df.set_meta(
tsdata.apply(
                pyam.cumulative,
                raw=False,
                axis=1,
                first_year=2020,
                last_year=2030
            ),
            label1
        )
        meta_docs[label1] = f'Cumulative {indi} from 2020 until 2030 (including the last year, {cumulative_unit}) ({variable})'
        
        df.set_meta(
            tsdata.apply(
                pyam.cumulative,
                raw=False,
                axis=1,
                first_year=2030,
                last_year=lastyear
            ),
            label2
        )
        meta_docs[label2] = f'Cumulative {indi} from 2030 until {lastyear} (including the last year, {cumulative_unit}) ({variable})'


#%% GHG* % reduction compared to 1990 (incl int transport) added Fabio
ghg1990 = 4790.123 # Total net emissions with int Transport EEA Source: https://www.eea.europa.eu/data-and-maps/data/data-viewers/greenhouse-gases-viewer) 
base_year_ghg = 1990
last_years = [2020, 2025, 2030, 2035, 2040, 2050]
for last_year in last_years:
    name = f'GHG* emissions reductions {base_year_ghg}-{last_year} %'
    values = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (AR4) (EEA)').timeseries()
    reduction = 100* (1-(values[last_year] / ghg1990))
    df.set_meta(reduction, name, )



#%% GHG** % reduction compared to 1990 (incl int transport)  (intra EU)

ghg1990 = 4790.123 -99.40 # 
# 4790 is Total net emissions with int Transport EEA Source: https://www.eea.europa.eu/data-and-maps/data/data-viewers/greenhouse-gases-viewer)
# NOTE 99.40 Is extra-eu bunkers (to be subtracted as we want to include only in intra-eu bunkers). 
# Extra eu shipping (99.40) was calculated based on data from ESABCC in 1990: 156.66 (Total) -56.27 (intra-eu domestic bunkers )=99.40

base_year_ghg = 1990
last_years = [2020, 2025, 2030, 2035, 2040, 2050]

for last_year in last_years:
    name = f'GHG** emissions reductions {base_year_ghg}-{last_year} %'
    values = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)').timeseries()
    reduction = 100* (1-(values[last_year] / ghg1990))
    df.set_meta(reduction, name, )


#%% GHG** % reduction compared to 2030 (incl int transport)

base_year_ghg = 2030
last_years = [2035, 2040, 2045, 2050]

for last_year in last_years:
    name = f'GHG** emissions reductions {base_year_ghg}-{last_year} %'
    values = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)').timeseries()
    reduction = 100* (1-(values[last_year] / values[base_year_ghg]))
    df.set_meta(reduction, name, )


#%% GHG* % reduction compared to 2019 incl. International transport

ghg2019 = 3634.836 # added Fabio -> Total net emissions with int Transport EEA Source: https://www.eea.europa.eu/data-and-maps/data/data-viewers/greenhouse-gases-viewer) 
base_year_ghg = 2019
last_years = [2030, 2035, 2040, 2050]

for last_year in last_years:
    name = f'GHG* emissions reductions {base_year_ghg}-{last_year} %' 
    values = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (AR4) (EEA)').timeseries() 
    reduction = 100* (1-(values[last_year] / ghg2019))
    df.set_meta(reduction, name, )

#%% Non-CO2 % reduction 2020-2050

base_year_ghg = 2020
last_years = [2030, 2050]

for last_year in last_years:
    name = f'Non-CO2 emissions reductions {base_year_ghg}-{last_year} %'
    values = df.filter(variable=f'{prefix}Emissions|Total Non-CO2').timeseries()
    reduction = 100* (1-(values[last_year] / values[base_year_ghg]))
    df.set_meta(reduction, name, )
    

# =============================================================================
#%% Calculation of indicator variables
# =============================================================================
ynz_variables = []

# =============================================================================
# Emissions
# =============================================================================
indis_add = [
            f'{prefix}Emissions|CO2',
            f'{prefix}Emissions|Total Non-CO2',
            f'{prefix}Emissions|Kyoto Gases (AR4) (EEA)', 
            f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)',
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

ynz_variables.append(f'{prefix}Primary Energy|Biomass')

# =============================================================================
# Primary energy - Renewables share
    # variable not in Harmonized list
name = 'Primary Energy|Renewables (incl.Biomass)|Share'
df.divide(f'{prefix}Primary Energy|Renewables (incl.Biomass)', 
          f'{prefix}Primary Energy',
          name, 
          ignore_units='-',
          append=True)
ynz_variables.append(name)


# Non-Biomass Renewables
name = 'Primary Energy|Non-Biomass Renewables|Share'
df.divide(f'{prefix}Primary Energy|Non-Biomass Renewables', 
          f'{prefix}Primary Energy',
          name, 
          ignore_units='-',
          append=True)
ynz_variables.append(name)

# =============================================================================
# Primary energy - Fossil share

name = 'Primary Energy|Fossil|Share'
df.divide(f'{prefix}Primary Energy|Fossil', 
          f'{prefix}Primary Energy',
          name, 
          ignore_units='-',
          append=True)
ynz_variables.append(name)


name = 'Primary Energy|Fossil|w/o CCS|Share'
df.divide(f'{prefix}Primary Energy|Fossil|w/o CCS', 
          f'{prefix}Primary Energy',
          name, 
          ignore_units='-',
          append=True)
ynz_variables.append(name)


# =============================================================================
# Secondary energy electricity renewables & hydrogen
# =============================================================================

rencomps = [
      f'{prefix}Secondary Energy|Electricity|Biomass',
      f'{prefix}Secondary Energy|Electricity|Geothermal',
      f'{prefix}Secondary Energy|Electricity|Hydro',
      f'{prefix}Secondary Energy|Electricity|Solar',
      f'{prefix}Secondary Energy|Electricity|Wind',]

df.aggregate(f'{prefix}Secondary Energy|Electricity|Renewables (incl.Biomass)', 
                components=rencomps, append=True)
    

# % of renewables in electricity
nv = f'Secondary Energy|Electricity|Renewables (incl.Biomass)|Share'
df.divide(f'{prefix}Secondary Energy|Electricity|Renewables (incl.Biomass)', 
          f'{prefix}Secondary Energy|Electricity', 
          nv,
          ignore_units='-',
          append=True)
ynz_variables.append(nv)

# % of non-bio renewables in electricity
nv = f'Secondary Energy|Electricity|Non-Biomass Renewables|Share'
df.divide(f'{prefix}Secondary Energy|Electricity|Non-Biomass Renewables', 
          f'{prefix}Secondary Energy|Electricity',
          nv,
          ignore_units='-',
          append=True)
ynz_variables.append(nv)

# =============================================================================
#Hydrogen production as share of FE *** Need to check
name = 'Hydrogen production|Final Energy|Share'
df.divide(f'Secondary Energy|Hydrogen', 'Final Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)

# =============================================================================
# =============================================================================
# # Final energy
# Since Final Energy not harmonzied - this can be written into original variable structure

ynz_variables.append('Final Energy')
# =============================================================================
# #% of final energy that is electrified

nv = 'Final Energy|Electrification|Share'
nu = '-'
df.divide('Final Energy|Electricity', 'Final Energy', 
          nv, 
          ignore_units=nu, 
          append=True)
    
ynz_variables.append(nv)
# 
# =============================================================================
# Sectoral final energy fossil shares
# =============================================================================

# =============================================================================
    # # Industry
nv = 'Final Energy|Industry|Fossil|Share'
nu = '-'
df.divide('Final Energy|Industry|Fossil', 'Final Energy|Industry', 
          nv, 
          ignore_units=nu, 
          append=True)
    
ynz_variables.append(nv)
    
    # # =============================================================================
    # # Residential and Commercial
nv = 'Final Energy|Residential and Commercial|Fossil|Share'
nu = '-'
df.divide('Final Energy|Residential and Commercial|Fossil', 'Final Energy|Residential and Commercial', 
          nv, 
          ignore_units=nu, 
          append=True)
    
ynz_variables.append(nv)
    
    # # =============================================================================
    # # Transportation
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
# Where derived from Harmonized Primary Energy - probably this should go into diagnostics
# =============================================================================
# Fossil fuel import dependency
nv = f'Primary Energy|Trade|Share'
df.divide('Trade', f'{prefix}Primary Energy',
          nv,
          ignore_units='-',
          append=True)
    
df.multiply(nv, -100, 'Primary Energy|Import Dependency', ignore_units='%', append=True)
ynz_variables.append('Primary Energy|Import Dependency')
    
    
nv = f'Primary Energy|Fossil|Trade|Share'
df.divide('Trade|Primary Energy|Fossil', f'{prefix}Primary Energy',
          nv,
          ignore_units='-',
          append=True)
df.multiply(nv, -100, 'PE Import dependency|Fossil', ignore_units='%', append=True)
ynz_variables.append('PE Import dependency|Fossil')
    
    
nv = 'Trade|Fossil|Share'  # This variable calculated from variables that aren't harmonized, so not a diagnostic
df.divide('Trade|Primary Energy|Fossil', 'Trade',
          nv,
          ignore_units='-',
          append=True)
ynz_variables.append(nv)

ynz_variables.append('Trade')
ynz_variables.append('Trade|Primary Energy|Fossil')

# dsfdsfs
df.convert_unit('-', '%', factor=100, inplace=True)

# =============================================================================
#%% Calculate indicators in year of net-zero and 2050
# =============================================================================

df.interpolate(time=range(2005,2070), inplace=True)


for v in ynz_variables:
    print(v)
    datats = filter_and_convert(df, v)
    nu = datats.reset_index().unit.unique()[0]
    
    name = f'{v} in year of net zero, {nu}'
    df.set_meta(datats.apply(lambda x: x[get_from_meta_column(df, x,
                                                              nameNZCO2)],
                                            raw=False, axis=1), name)    
    if v==f'{prefix}Emissions|Kyoto Gases (incl. indirect AFOLU)':
        name = f'{v} in 2025, {nu}'
        df.set_meta_from_data(name, variable=v, year=2025)
        name = f'{v} in 2030, {nu}'
        df.set_meta_from_data(name, variable=v, year=2030)
    elif v==f'{prefix}Emissions|Kyoto Gases (AR4) (EEA)': # added Fabio
        name = f'{v} in 2025, {nu}'
        df.set_meta_from_data(name, variable=v, year=2025)
        name = f'{v} in 2030, {nu}'
        df.set_meta_from_data(name, variable=v, year=2030)
        
    name = f'{v} in 2050, {nu}'
    df.set_meta_from_data(name, variable=v, year=2050)
    
df.set_meta_from_data('CCUS in 2050, Mt CO2/yr', 
                      variable='Carbon Capture, Use, and Sequestration', 
                      year=2050)
df.set_meta_from_data('CCUS in 2070, Mt CO2/yr', 
                      variable='Carbon Capture, Use, and Sequestration', 
                      year=2070)

name = 'Emissions|CO2|LULUCF Direct+Indirect in 2050, Mt CO2/yr'
df.set_meta_from_data(name=name, 
                      variable=f'{prefix}Emissions|CO2|LULUCF Direct+Indirect', 
                      year=2050, 
                      region='EU27')

# =============================================================================
#%% Additional filter based on GHG emissions resduction # Fabio added additional filters
# =============================================================================

def set_meta_from_condition(df, name, condition, index=None):
    data = np.where(condition, True, False)
    if index is not None:
        data = np.where(index, data, False)
    df.set_meta(data, name=name)

base_year = 1990
target2030 = 55  # 55% reduction
target2050 = 300  # 300 Mt in 2050

# Pass based on GHG* emissions reductions
name = f'Pass based on GHG* emissions reductions'
keep_2030 = df.meta["GHG* emissions reductions 1990-2030 %"]
keep_2050 = df.meta[f'{prefix}Emissions|Kyoto Gases (AR4) (EEA) in 2050, Mt CO2-equiv/yr']
condition = (keep_2030 >= target2030) & (keep_2050 <= target2050)
set_meta_from_condition(df, name, condition)

# Pass based on GHG** emissions reductions
name = f'Pass based on GHG** emissions reductions'
keep_2030 = df.meta["GHG** emissions reductions 1990-2030 %"]
keep_2050 = df.meta[f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only) in 2050, Mt CO2-equiv/yr']
condition = (keep_2030 >= target2030) & (keep_2050 <= target2050)
set_meta_from_condition(df, name, condition)

# Pass based on positive Non-CO2 emissions
name = f'Pass based on positive Non-CO2 emissions'
keep_2050 = df.meta[f'{prefix}Emissions|Total Non-CO2 in 2050, Mt CO2-equiv/yr']
condition = keep_2050 > 0
set_meta_from_condition(df, name, condition)

# CCUS in 2050
name = 'CCUS in 2050, Mt CO2/yr'
name1 = 'CCUS < 425 Mt CO2 in 2050'
set_meta_from_condition(df, name1, df.meta[name] < 425)
name1 = 'CCUS < 500 Mt CO2 in 2050'
set_meta_from_condition(df, name1, df.meta[name] < 500)

# Primary Energy|Biomass in 2050
name = 'Primary Energy|Biomass in 2050, EJ/yr'
set_meta_from_condition(df, name, True, index=df.index)

name1 = 'Primary Energy|Biomass <9 EJ/yr in 2050'
set_meta_from_condition(df, name1, df.meta[name] < 9)

# Emissions|CO2|LULUCF Direct+Indirect in 2050
name = 'Emissions|CO2|LULUCF Direct+Indirect in 2050, Mt CO2/yr'
name1 = 'Emissions|CO2|LULUCF Direct+Indirect <400 Mt CO2/yr in 2050'
set_meta_from_condition(df, name1, df.meta[name] < 400)

# =============================================================================
#%% Write out  meta sheet with formatting
# =============================================================================
# To reproduce the original excel sheet - we need to read in the specific 
# column names and order
dfoc = pd.read_csv('meta_columns.csv')

pre_cols = ['version',
             'Source',
             'Reference',
             'Doi',
             'Vetting status',
             'Category name',
             'Category',
             'Coverage: R5',
             'Coverage: global',
             'Feasibility Flag|Overall',
             'Feasibility Flag|Hydrogen',
             'Feasibility Flag|Biomass',
             'Feasibility Flag|Final Energy',
             'Feasibility Flag|CCUS',
             ]

all_cols = pre_cols + dfoc[18:]['0'].tolist()

df.meta = df.meta[all_cols]

# Write out
write_meta_sheet(df, fn_out, startfile=True)


#%% Compare the dataframes

# Load the original
df_o = pyam.IamDataFrame('C:/Users/byers/IIASA/ECE.prog - Documents/Projects/EUAB/iconics/20230512/iconics_NZ_data_and_table_20230512_v17.xlsx')

df_orig = df_o.rename(
    {'variable':
     {'Primary Energy|Non-biomass renewables|Share':
      'Primary Energy|Non-Biomass Renewables|Share',
      'CCUS': 'Carbon Capture, Use, and Sequestration'}})
    
df_orig.filter(variable=new_vars, 
               year=years[4:], 
               inplace=True)

df_orig.filter(variable='Final Energy|Industry|Electricity|Share', 
            keep=False, 
            inplace=True)



df_new = df.filter(variable=new_vars, 
                   year=years[4:])
df_new.rename({'variable':{
                        'Primary Energy|Import Dependency':
                        'PE Import dependency'}}, 
                                  inplace=True)


dfc = pyam.compare(df_orig, df_new,
             left_label='orig',
             right_label='reprod')

dfc = dfc.reset_index()
if len(dfc)>0:
    print('Warning: possible differences detected')
    dfc.to_excel('compare_calculated_variables.xlsx')
    os.startfile('compare_calculated_variables.xlsx')


