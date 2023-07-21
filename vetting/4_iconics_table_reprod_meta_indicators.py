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


user = 'byers'

output_folder = f'outputs\\'

instance = 'eu-climate-advisory-board'

datestr = '20230712'  

fn_out = f'{output_folder}iconics_NZ_data_and_table_{datestr}_v17_reprod_meta.xlsx'


#%% Load data
    # vetting = pd.read_excel(wbstr, sheet_name='Vetting_flags')


years = list(range(2015,2071))

prefix = 'Diagnostics|Harmonized|'

varlist = ['Diagnostics*',
           'Final Energy*',
           'Trade|*',
           'Imports*',
           'Carbon Sequestration*',
           'Secondary Energy|Hydrogen',
           'Carbon Capture|Usage', 
           'Carbon Capture|Storage',
           ]


# New variables
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


#% load pyam data
dfin = pyam.read_iiasa(instance,
                        # model=models,
                        # scenario=scenarios,
                        variable=varlist,
                        year=years,
                        region='EU27', 
                        meta=True)


# For testing - drop all data with "Share"
dfin.filter(variable='*Share', keep=False, inplace=True)



# Keep only the trade data in 'EJ/yr'
dfin.filter(variable='Trade*', unit='billion US$2010/yr', 
            keep=False, inplace=True)


# All commented out (Ed refactor, 20 lines)
    # df_remFETF  = pyam.IamDataFrame(f'{main_folder}from_FabioS\\2023_05_12\\AdvisoryBoard_REMIND3p2_additional_variables.xlsx')
    # df_remFETF.filter(variable='Final Energy|Transportation|Liquids|Fossil', inplace=True)
    
    # dfin.rename({'model':{'AIM_CGE 2.2': 'AIM/CGE 2.2'}}, inplace=True)
    
    # ngfs_rename =   {'d_delfrag': 'NGFS-Delayed transition',
    #                  'd_rap': 'NGFS-Divergent Net Zero',
    #                  'h_cpol': 'NGFS-Current Policies',
    #                  'h_ndc': 'NGFS-Nationally Determined Contributions (NDCs)',
    #                  'o_1p5c': 'NGFS-Net Zero 2050',
    #                  'o_2c': 'NGFS-Below 2?C',
    #                       }
    
    # dfin.rename({'scenario':ngfs_rename}, inplace=True)
    
    # dfin.append(df_remFETF, inplace=True)
    
    # dfin.load_meta(wbstr, sheet_name='Vetting_flags')   


#%% Filter scenarios (RESTART FROM HERE) 
# =============================================================================

df = dfin.filter(region='EU27')
# df.interpolate(time=range(2000,2101), inplace=True)

# Filter years in case of odd zeros
years = list(range(1990,2016,5))+[2019]+list(range(2020,2071,5))
df.filter(year=years, inplace=True)




    # vetting & climate
    # df.filter(OVERALL_binary='PASS', inplace=True)
df.filter(**{'Vetting status':'pass'}, inplace=True)


    # df.meta.loc[df.meta['OVERALL_Assessment']=='Regional only', 'Category'] = 'Regional only'
df.filter(Category=['C1*', 'Regional only'], inplace=True)


    # Drop GCAM Current Policies scenarios
    # df.filter(model='GCAM*', scenario='*CurPol*', keep=False, inplace=True)

    # Remove (almost) duplicate REMIND 3.2 scenarios
    # remkeep = df.filter(model='REMIND 3.2', scenario='*_withICEPhOP*',)
    # df.filter(model='REMIND 3.2', keep=False, inplace=True)
    # df.append(remkeep, inplace=True)

meta_docs = {}

#%% check aggregates  # Fails....
    # comps = [f'{prefix}Emissions|CO2|Energy and Industrial Processes',
    #          f'{prefix}Emissions|CO2|LULUCF Direct+Indirect',
    #          f'{prefix}Emissions|Total Non-CO2']
    # df.convert_unit(
    #     'Mt CO2/yr','Mt CO2-equiv/yr').check_aggregate(
    #         f'{prefix}Emissions|Kyoto Gases (incl. indirect AFOLU)', components=comps)
# =============================================================================
    # #%% Variable Aggregations (skip this whole section?)
    # # =============================================================================
    
    # # 'Emissions|CO2'
    # a = f'{prefix}Emissions|CO2|Industrial Processes'
    # b = df.filter(variable=a).convert_unit('Mt CO2-equiv/yr', 'Mt CO2/yr')
    # df.filter(variable=a, keep=False, inplace=True)
    # df.append(b, inplace=True)
    
    
    # # Carbon Sequestration|CCS
    # # Rename other ECEMF variables, in case missing
    # ccs_rename = {'variable':{'Carbon Capture|Storage|Biomass':'Carbon Sequestration|CCS|Biomass',
    #           'Carbon Capture|Storage|Direct Air Capture': 'Carbon Sequestration|Direct Air Capture',
    #           'Carbon Capture|Storage|Fossil': 'Carbon Sequestration|CCS|Fossil'}}
    
    # dfwit = df.filter(model='WITCH*', variable=['Carbon Capture|Storage|Biomass',
    #                                             'Carbon Capture|Storage|Direct Air Capture',
    #                                             'Carbon Capture|Storage|Fossil'])#.
    # dfwit.rename(ccs_rename, inplace=True)
    # df.append(dfwit, inplace=True)
    
    
    # # Aggregate CCS in scenarios where missing
    # components = ['Carbon Sequestration|CCS|Industrial Processes',
    #               'Carbon Sequestration|CCS|Fossil',
    #               'Carbon Sequestration|CCS|Biomass']
    # aggregate_missing_only(df, 'Carbon Sequestration|CCS', 
    #                        components=components, 
    #                        append=True)
    
    
    # # =============================================================================
    # # # For REMIND 2.1 / 3,2 scenarios
    # # =============================================================================
    # # This is done due to different reporting by these REMIND scenarios
df.aggregate('Carbon Capture, Use, and Sequestration', components=['Carbon Capture|Usage', 'Carbon Capture|Storage'],
                          append=True)
    
    # # others - assume CCS is right and add DACCS
dfo = df.filter(model=['REMIND 2.1', 'REMIND 3.2'], keep=False)
dfo.aggregate('Carbon Capture, Use, and Sequestration',['Carbon Sequestration|CCS', 'Carbon Sequestration|Direct Air Capture'],  append=True)
df.append(dfo.filter(variable='Carbon Capture, Use, and Sequestration'), inplace=True)
    
    
    
    # ## Fossil fuels
    
    # components = ['Primary Energy|Coal', 'Primary Energy|Gas', 'Primary Energy|Oil']
    # aggregate_missing_only(df, 'Primary Energy|Fossil', components, append=True)
    
    # components = ['Primary Energy|Coal|w/o CCS','Primary Energy|Oil|w/o CCS','Primary Energy|Gas|w/o CCS']
    # aggregate_missing_only(df, 'Primary Energy|Fossil|w/o CCS', components, append=True)
    
    
# Primary Energy renewables
components = [f'{prefix}Primary Energy|Biomass', 
              f'{prefix}Primary Energy|Geothermal',
              f'{prefix}Primary Energy|Hydro', 
              f'{prefix}Primary Energy|Solar',
              f'{prefix}Primary Energy|Wind',
              ]
    
name = f'{prefix}Primary Energy|Renewables (incl.Biomass)'
df.aggregate(name, components, append=True)
    # name = 'Primary Energy|Non-biomass renewables'
    # df.aggregate(name, components[1:], append=True)
    
    
# Final energy  fossil sectoral
# Already in? (71 scenarios)
components = ['Final Energy|Industry|Gases|Coal',
              'Final Energy|Industry|Gases|Natural Gas',
              'Final Energy|Industry|Liquids|Coal',
              'Final Energy|Industry|Liquids|Gas',
              'Final Energy|Industry|Liquids|Oil',
              'Final Energy|Industry|Solids|Coal',
              ]
name = 'Final Energy|Industry|Fossil'
# df.aggregate(name, components=components, append=True)
aggregate_missing_only(df, name, components, append=True)
    
# Res & Comm
# Already in? (71 scenarios)
components = ['Final Energy|Residential and Commercial|Gases|Coal',
              'Final Energy|Residential and Commercial|Gases|Natural Gas',
              'Final Energy|Residential and Commercial|Liquids|Coal',
              'Final Energy|Residential and Commercial|Liquids|Gas',
              'Final Energy|Residential and Commercial|Liquids|Oil',
              'Final Energy|Residential and Commercial|Solids|Coal',
              ]
name = 'Final Energy|Residential and Commercial|Fossil'
# df.aggregate(name, components=components, append=True)
aggregate_missing_only(df, name, components, append=True)
    
# Transportation
# Already in? (76 scenarios)
components = ['Final Energy|Transportation|Gases|Coal',
              'Final Energy|Transportation|Gases|Natural Gas',
              'Final Energy|Transportation|Liquids|Coal',
              'Final Energy|Transportation|Liquids|Gas',
              'Final Energy|Transportation|Liquids|Oil',
              ]
name = 'Final Energy|Transportation|Fossil'
# df.aggregate(name, components=components, append=True)
aggregate_missing_only(df, name, components, append=True)
    
                     
    # # Aggregate
    #     # Already in?
    #     # components = ['Final Energy|Transportation|Gases|Fossil', 'Final Energy|Transportation|Liquids|Fossil']
    #     # rem_FETF = df.filter(model=['REMIND 3.2', 'REMIND 2.1'], ).aggregate(name, components=components)
    #     # df.append(rem_FETF, inplace=True)
    
    
    
# Trade
# df.filter(variable='Trade', keep=False, inplace=True)
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
# df.aggregate(name, components=components, append=True)
aggregate_missing_only(df, name, components, append=True)

#=============================================================================
#%% Year of netzero 
# =============================================================================


# Calculate year of net zeros
specdic = {
    'CO2': {
        'variable': f'{prefix}Emissions|CO2',
        'unitin': 'Mt CO2/yr',
        'unitout': 'Gt CO2/yr',
        'factor': 0.001
    },
           # 'GHGs full':{'variable': f'{prefix}Emissions|Kyoto Gases (incl. indirect AFOLU)',
           #                        'unitin': 'Mt CO2-equiv/yr',
           #                        'unitout': 'Gt CO2-equiv/yr',
           #                        'factor': 0.001},
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
    },
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

# df.interpolate(time=range(2000,2101), inplace=True)


#%% Cumulative calcs


specdic = {'net CO2': {'variable': f'{prefix}Emissions|CO2',
                       'unitin': 'Mt CO2/yr',
                       'unitout': 'Gt CO2/yr',
                       'factor': 0.001},
           'Non-CO2':{'variable': f'{prefix}Emissions|Total Non-CO2',
                                  'unitin': 'Mt CO2-equiv/yr',
                                  'unitout': 'Gt CO2-equiv/yr',
                                  'factor': 0.001},            
           'CCS':{'variable': f'Carbon Sequestration|CCS',
                                  'unitin': 'Mt CO2/yr',
                                  'unitout': 'Gt CO2/yr',
                                  'factor': 0.001},
           'BECCS':{'variable': f'Carbon Sequestration|CCS|Biomass',
                                  'unitin': 'Mt CO2/yr',
                                  'unitout': 'Gt CO2/yr',
                                  'factor': 0.001},
               # 'GHGs (incl. indirect AFOLU)':{'variable': f'{prefix}Emissions|Kyoto Gases (incl. indirect AFOLU)',
               #                        'unitin': 'Mt CO2-equiv/yr',
               #                        'unitout': 'Gt CO2-equiv/yr',
               #                        'factor': 0.001},
            'GHGs* (incl. indirect AFOLU)':{'variable': f'{prefix}Emissions|Kyoto Gases (AR4) (EEA)',
                                  'unitin': 'Mt CO2-equiv/yr',
                                  'unitout': 'Gt CO2-equiv/yr',
                                  'factor': 0.001},
            'GHGs** (incl. indirect AFOLU)':{'variable': f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)',
                                  'unitin': 'Mt CO2-equiv/yr',
                                  'unitout': 'Gt CO2-equiv/yr',
                                  'factor': 0.001},
            'CO2 FFI':{'variable': f'{prefix}Emissions|CO2|Energy and Industrial Processes', 
                                  'unitin': 'Mt CO2-equiv/yr',
                                  'unitout': 'Gt CO2-equiv/yr',
                                  'factor': 0.001},
            'AFOLU (direct+indirect)': {'variable': f'{prefix}Emissions|CO2|LULUCF Direct+Indirect',
                       'unitin': 'Mt CO2/yr',
                       'unitout': 'Gt CO2/yr',
                       'factor': 0.001},
           }
           

# to 2050
# baseyear = 2020
# lastyear = 2050

# for indi, config in specdic.items():
#     variable = config['variable']
#     tsdata = filter_and_convert(df, variable, unitin=config['unitin'], unitout=config['unitout'], factor=config['factor'])
#     cumulative_unit = config['unitout'].split('/yr')[0]
    
#     # to year of net-zero CO2
#     name = f'cumulative {indi} ({baseyear} to year of net zero CO2, {cumulative_unit})'
#     df.set_meta(tsdata.apply(lambda x: pyam.cumulative(x, first_year=baseyear, last_year=get_from_meta_column(df, x,                                                                nameNZCO2)), raw=False, axis=1), name)
#     meta_docs[name] = f'Cumulative {indi} from {baseyear} until year of net zero CO2 (including the last year, {cumulative_unit}) ({variable})'
    
#     # to 2050
#     label = f'cumulative {indi} ({baseyear}-{lastyear}, {cumulative_unit})'
#     df.set_meta(tsdata.apply(pyam.cumulative, raw=False, axis=1, first_year=baseyear, last_year=lastyear), label)
#     meta_docs[name] = f'Cumulative {indi} from {baseyear} until {lastyear} (including the last year, {cumulative_unit}) ({variable})'

#     # 2020 to 2030
#     if indi in ['net CO2', 'GHGs (incl. indirect AFOLU)', 'GHGs* (incl. indirect AFOLU)', 'GHGs** (incl. indirect AFOLU)',
#                 'AFOLU (direct+indirect)']:
#         label = f'cumulative {indi} (2020-2030, {cumulative_unit})'
#         df.set_meta(tsdata.apply(pyam.cumulative, raw=False, axis=1, first_year=2020, last_year=2030), label)
#         meta_docs[name] = f'Cumulative {indi} from 2030 until {lastyear} (including the last year, {cumulative_unit}) ({variable})'    
    
#     # 2030 to 2050
#     if indi in ['net CO2', 'GHGs (incl. indirect AFOLU)', 'GHGs* (incl. indirect AFOLU)', 'GHGs** (incl. indirect AFOLU)',
#                 'AFOLU (direct+indirect)']:
#         label = f'cumulative {indi} (2030-{lastyear}, {cumulative_unit})'
#         df.set_meta(tsdata.apply(pyam.cumulative, raw=False, axis=1, first_year=2030, last_year=lastyear), label)
#         meta_docs[name] = f'Cumulative {indi} from 2030 until {lastyear} (including the last year, {cumulative_unit}) ({variable})'    


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



#%% GHG % reduction compared to 1990
    # ghg1990 = 4633 # NOTE: without int. transport -> `Emissions|Kyoto Gases (AR4) (UNFCCC)`
    # base_year_ghg = 1990
    # last_years = [2020, 2025, 2030, 2035, 2040, 2050]
    # for last_year in last_years:
    #     name = f'GHG emissions reductions {base_year_ghg}-{last_year} %'
    #     a = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (incl. indirect AFOLU)').timeseries()
    #     rd = 100* (1-(a[last_year] / ghg1990))
    #     df.set_meta(rd, name, )

#%% GHG* % reduction compared to 1990 (incl int transport) added Fabio
ghg1990 = 4790.123 # added Fabio -> Total net emissions with int Transport EEA Source: https://www.eea.europa.eu/data-and-maps/data/data-viewers/greenhouse-gases-viewer) 
base_year_ghg = 1990
last_years = [2020, 2025, 2030, 2035, 2040, 2050]
for last_year in last_years:
    name = f'GHG* emissions reductions {base_year_ghg}-{last_year} %'
    a = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (AR4) (EEA)').timeseries()
    rd = 100* (1-(a[last_year] / ghg1990))
    df.set_meta(rd, name, )



#%% GHG** % reduction compared to 1990 (incl int transport) added Fabio  (intra EU)
ghg1990 = 4790.123 -99.40 # added Fabio -> 
# 4790 is Total net emissions with int Transport EEA Source: https://www.eea.europa.eu/data-and-maps/data/data-viewers/greenhouse-gases-viewer)
# NOTE 99.40 Is extra-eu bunkers (to be subtracted as we want to include only in intra-eu bunkers). 
# Extra eu shipping (99.40) was calculated based on data from ESABCC in 1990: 156.66 (Total) -56.27 (intra-eu domestic bunkers )=99.40
base_year_ghg = 1990
last_years = [2020, 2025, 2030, 2035, 2040, 2050]
for last_year in last_years:
    name = f'GHG** emissions reductions {base_year_ghg}-{last_year} %'
    a = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)').timeseries()
    rd = 100* (1-(a[last_year] / ghg1990))
    df.set_meta(rd, name, )




#%% GHG** % reduction compared to 2030 (incl int transport) added Fabio 20230421
base_year_ghg = 2030
last_years = [2035, 2040, 2045, 2050]
for last_year in last_years:
    name = f'GHG** emissions reductions {base_year_ghg}-{last_year} %'
    a = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)').timeseries()
    rd = 100* (1-(a[last_year] / a[base_year_ghg]))
    df.set_meta(rd, name, )



#%% GHG % reduction compared to2019
    # ghg2019 = 3364 # NOTE: without int. transport -> `Emissions|Kyoto Gases (AR4) (UNFCCC)`
    # base_year_ghg = 2019
    # last_years = [2030, 2035, 2040, 2050]
    # for last_year in last_years:
    #     name = f'GHG emissions reductions {base_year_ghg}-{last_year} %'
    #     a = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (incl. indirect AFOLU)').timeseries()
    #     rd = 100* (1-(a[last_year] / ghg2019))
    #     df.set_meta(rd, name, )


#%% GHG* % reduction compared to2019 incl. International transport added Fabio
ghg2019 = 3634.836 # added Fabio -> Total net emissions with int Transport EEA Source: https://www.eea.europa.eu/data-and-maps/data/data-viewers/greenhouse-gases-viewer) 
base_year_ghg = 2019
last_years = [2030, 2035, 2040, 2050]
for last_year in last_years:
    name = f'GHG* emissions reductions {base_year_ghg}-{last_year} %' 
    a = df.filter(variable=f'{prefix}Emissions|Kyoto Gases (AR4) (EEA)').timeseries() 
    rd = 100* (1-(a[last_year] / ghg2019))
    df.set_meta(rd, name, )

# =============================================================================
#%% Non-CO2 % reduction 2020-2050
# =============================================================================
base_year = 2020
last_years = [2030, 2050]
for last_year in last_years:
    name = f'Non-CO2 emissions reductions {base_year}-{last_year} %'
    a = df.filter(variable=f'{prefix}Emissions|Total Non-CO2').timeseries()
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
            f'{prefix}Emissions|CO2',
            f'{prefix}Emissions|Total Non-CO2',
            # f'{prefix}Emissions|Kyoto Gases (incl. indirect AFOLU)',
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
df.divide(f'{prefix}Primary Energy|Renewables (incl.Biomass)', f'{prefix}Primary Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)


# Note biomass and Biomass
name = 'Primary Energy|Non-Biomass Renewables|Share'
df.divide(f'{prefix}Primary Energy|Non-Biomass Renewables', f'{prefix}Primary Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)

# =============================================================================
# Primary energy - Fossil share

name = 'Primary Energy|Fossil|Share'
df.divide(f'{prefix}Primary Energy|Fossil', f'{prefix}Primary Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)


name = 'Primary Energy|Fossil|w/o CCS|Share'
df.divide(f'{prefix}Primary Energy|Fossil|w/o CCS', f'{prefix}Primary Energy',
          name, ignore_units='-',
          append=True)
ynz_variables.append(name)


# =============================================================================
# Secondary energy electricity renewables & hydrogen
# =============================================================================

# Secondary energy Renewables (not needed if data already in place?)
    # Drop non-bio renewables
    # df.filter(variable=f'{prefix}Secondary Energy|Electricity|Renewables (incl.Biomass)', 
    #               keep=False, inplace=True)
    
    # df.filter(variable=f'{prefix}Secondary Energy|Electricity|Non-Biomass Renewables', 
    #               keep=False, inplace=True)
    
rencomps = [
      f'{prefix}Secondary Energy|Electricity|Biomass',
      f'{prefix}Secondary Energy|Electricity|Geothermal',
      f'{prefix}Secondary Energy|Electricity|Hydro',
      f'{prefix}Secondary Energy|Electricity|Solar',
      f'{prefix}Secondary Energy|Electricity|Wind',]

df.aggregate(f'{prefix}Secondary Energy|Electricity|Renewables (incl.Biomass)', 
                components=rencomps, append=True)
    
    # df.aggregate(f'{prefix}Secondary Energy|Electricity|Non-Biomass Renewables', 
    #                 components=rencomps[1:], append=True)


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

# =============================================================================
# =============================================================================

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
    
df.set_meta_from_data('CCUS in 2050, Mt CO2/yr', variable='Carbon Capture, Use, and Sequestration', year=2050)
df.set_meta_from_data('CCUS in 2070, Mt CO2/yr', variable='Carbon Capture, Use, and Sequestration', year=2070)    
name = 'Emissions|CO2|LULUCF Direct+Indirect in 2050, Mt CO2/yr'
df.set_meta_from_data(name=name, variable=f'{prefix}Emissions|CO2|LULUCF Direct+Indirect', year=2050, region='EU27')
#%% Calculate EU share of global emissions, 2040 and 2020-2050.

# instance = 'eu-climate-advisory-board-internal'

# x = df.meta.reset_index()[['model','scenario']]
# msdic = {k: list(v) for k,v in x.groupby("model")["scenario"]}


# dfgc = df.filter(model='abc')
# for model, scenarios in msdic.items():
#     dfgc.append(pyam.read_iiasa(instance,
#                             model=model,
#                             scenario=scenarios,
#                             variable='Emissions|CO2',
#                             region='World', meta=False),
#                 inplace=True)

# dfgc.filter(year=list(range(2020,2051,5)), inplace=True)
# df.append(dfgc, inplace=True, ignore_meta_conflict=True)


# for year in [2030, 2040, 2050]:
#     dfd = df.filter(variable='Emissions|CO2', region=['EU27','World'], year=year)
    
#     dfd.divide('EU27', 'World', f'EU-share of World',
#                 axis='region', ignore_units='-',
#                 append=True)
    
#     df.append(dfd.filter(region='EU-share of World'), inplace=True)

#     df.set_meta_from_data(f'EU27-share of World Emissions|CO2 in {year}', 
#                           variable='Emissions|CO2',
#                           region=f'EU-share of World',
#                           year=year)



# =============================================================================
#%% Additional filter based on GHG emissions resduction # Fabio added additional filters
# =============================================================================
base_year = 1990
target2030 = 55  # 55% reduction
target2050 = 300  # 300 Mt in 2050

# name = f'Pass based on GHG emissions reductions'
# keep_2030 = df.meta["GHG emissions reductions 1990-2030 %"]
# keep_2050 = df.meta[f'{prefix}Emissions|Kyoto Gases (incl. indirect AFOLU) in 2050, Mt CO2-equiv/yr']
# keep_2030 = keep_2030[keep_2030>=target2030 ]
# keep_2050 = keep_2050[keep_2050<=target2050]
# index=keep_2030.index.intersection(keep_2050.index)
# df.set_meta(df.index.isin(index), name, )


name = f'Pass based on GHG* emissions reductions'
keep_2030 = df.meta["GHG* emissions reductions 1990-2030 %"]
keep_2050 = df.meta[f'{prefix}Emissions|Kyoto Gases (AR4) (EEA) in 2050, Mt CO2-equiv/yr']
keep_2030 = keep_2030[keep_2030>=target2030 ]
keep_2050 = keep_2050[keep_2050<=target2050]
index=keep_2030.index.intersection(keep_2050.index)
df.set_meta(df.index.isin(index), name, ) 


name = f'Pass based on GHG** emissions reductions'
keep_2030 = df.meta["GHG** emissions reductions 1990-2030 %"]
keep_2050 = df.meta[f'{prefix}Emissions|Kyoto Gases (AR4) (EEA - intra-EU only) in 2050, Mt CO2-equiv/yr']
keep_2030 = keep_2030[keep_2030>=target2030 ]
keep_2050 = keep_2050[keep_2050<=target2050]
index=keep_2030.index.intersection(keep_2050.index)
df.set_meta(df.index.isin(index), name, ) 

name = f'Pass based on positive Non-CO2 emissions'
keep_2050 = df.meta[f'{prefix}Emissions|Total Non-CO2 in 2050, Mt CO2-equiv/yr']
keep_2050 = keep_2050[keep_2050>0]
index=keep_2050.index
df.set_meta(df.index.isin(index), name, ) 

name = 'CCUS in 2050, Mt CO2/yr'
name1 = 'CCUS < 425 Mt CO2 in 2050'
df.meta[name1] = np.where(df.meta['CCUS in 2050, Mt CO2/yr']<425, True, False)
name1 = 'CCUS < 500 Mt CO2 in 2050'
df.meta[name1] = np.where(df.meta['CCUS in 2050, Mt CO2/yr']<500, True, False)

name = 'Primary Energy|Biomass in 2050, EJ/yr'
df.set_meta_from_data(name=name, variable='Primary Energy|Biomass', year=2050, region='EU27')
name1 = 'Primary Energy|Biomass <9 EJ/yr in 2050'
df.meta[name1] = np.where(df.meta[name]<9, True, False)

name = 'Emissions|CO2|LULUCF Direct+Indirect in 2050, Mt CO2/yr'
name1 = 'Emissions|CO2|LULUCF Direct+Indirect <400 Mt CO2/yr in 2050'
df.meta[name1] = np.where(df.meta[name]<400, True, False)



# =============================================================================
#%% Write out  meta sheet with formatting
# =============================================================================

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

write_meta_sheet(df, fn_out, startfile=True)


#%% Compare the dataframes


df_o = pyam.IamDataFrame('C:/Users/byers/IIASA/ECE.prog - Documents/Projects/EUAB/iconics/20230512/iconics_NZ_data_and_table_20230512_v17.xlsx')
#%%

df_orig = df_o.rename({'variable':{'Primary Energy|Non-biomass renewables|Share':
                            'Primary Energy|Non-Biomass Renewables|Share',
                            'CCUS': 'Carbon Capture, Use, and Sequestration'}})
df_orig.filter(variable=new_vars, year=years[4:], inplace=True)
df_orig.filter(variable='Final Energy|Industry|Electricity|Share', 
            keep=False, inplace=True)



df_new = df.filter(variable=new_vars, year=years[4:])
df_new.rename({'variable':{'Primary Energy|Import Dependency':
                           'PE Import dependency'}}, inplace=True)


dfc = pyam.compare(df_orig, df_new,
             left_label='orig',
             right_label='reprod')

dfc = dfc.reset_index()
if len(dfc)>0:
    print('Warning: possible differences detected')
    dfc.to_excel('compare_calculated_variables.xlsx')
    os.startfile('compare_calculated_variables.xlsx')




# =============================================================================
#%% Make the boxplots in the report
# =============================================================================

#%% iconics boxplots - only the passing 63 scenarios
df.meta['ghgfilter'] = df.meta[f'Pass based on GHG** emissions reductions']

dfb = df.meta
dfb = dfb.loc[dfb[f'Pass based on GHG** emissions reductions']==True]

if not os.path.exists(f'{output_folder}report_figures\\'):
    os.makedirs(f'{output_folder}report_figures\\')
    print("created folder : ", f'{output_folder}report_figures\\')



palette = 'GnBu'
#%% Cumulative emissions

fname = f'{output_folder}report_figures\\boxplot_cumEmissions.png'
varis = ['cumulative net CO2 (2020-2030, Gt CO2)',
        'cumulative net CO2 (2030-2050, Gt CO2)',
        'cumulative net CO2 (2020-2050, Gt CO2)',
        'cumulative Non-CO2 (2020-2050, Gt CO2-equiv)',
        'cumulative CCS (2020-2050, Gt CO2)',
        'cumulative BECCS (2020-2050, Gt CO2)',
        'cumulative GHGs** (incl. indirect AFOLU) (2020-2030, Gt CO2-equiv)',
        'cumulative GHGs** (incl. indirect AFOLU) (2030-2050, Gt CO2-equiv)',
        'cumulative GHGs** (incl. indirect AFOLU) (2020-2050, Gt CO2-equiv)',
        'cumulative AFOLU (direct+indirect) (2020-2050, Gt CO2)']
yticks = [v.strip('cumulative ') for v in varis]
yticks = [v.strip('net ') for v in yticks]
yticks = [v.replace(', Gt CO2)', ')') for v in yticks]
yticks = [v.replace(', Gt CO2-equiv)', ')') for v in yticks]


plot_box_meta(dfb, varis, yticks, xlabel='cumulative CO2 or GHGs, GtCO2e/yr',
              fname=fname, palette=palette)

#%% Emissions reductions

fname = f'{output_folder}report_figures\\boxplot_Emissions_reductions.png'
varis = [
        'GHG** emissions reductions 1990-2020 %',
        'GHG** emissions reductions 1990-2030 %',
        'GHG** emissions reductions 1990-2040 %',
        'GHG** emissions reductions 1990-2050 %',
        'Non-CO2 emissions reductions 2020-2030 %',
        'Non-CO2 emissions reductions 2020-2050 %',
        ]
yticks = [v.strip('cumulative ') for v in varis]
yticks = [v.strip('net ') for v in yticks]


plot_box_meta(dfb, varis, yticks, xlabel='% reduction',
              fname=fname, palette=palette)


#%% Emissions YNZ
fname = f'{output_folder}report_figures\\boxplot_emissions_YNZ.png'
varis = [
        'Emissions|Kyoto Gases (incl. indirect AFOLU) in year of net zero, Mt CO2-equiv/yr',
        'Emissions|Kyoto Gases (AR4) (EEA - intra-EU only) in year of net zero, Mt CO2-equiv/yr',
        'Emissions|CO2 in year of net zero, Mt CO2/yr',
        'Emissions|Total Non-CO2 in year of net zero, Mt CO2-equiv/yr',
        'Carbon Sequestration|CCS in year of net zero, Mt CO2/yr',
        'Carbon Sequestration|CCS|Biomass in year of net zero, Mt CO2/yr',
        'Carbon Sequestration|CCS|Fossil in year of net zero, Mt CO2/yr',
        'Carbon Sequestration|CCS|Industrial Processes in year of net zero, Mt CO2/yr',
        ]

yticks = [v.split(' in year')[0] for v in varis]
yticks = [v.split('|')[1:] for v in yticks]
yticks = [x[0] if len(x)==1 else x[0]+' '+ x[1] for x in yticks]

plot_box_meta(dfb, varis, yticks, xlabel='Emissions and removals in year of net-zero CO2',
              fname=fname, palette=palette)

#%% Emissions in 2050
fname = f'{output_folder}report_figures\\boxplot_emissions_2050.png'
varis = [
    'Emissions|Kyoto Gases (incl. indirect AFOLU) in 2050, Mt CO2-equiv/yr',
    'Emissions|Kyoto Gases (AR4) (EEA - intra-EU only) in 2050, Mt CO2-equiv/yr',           
    'Emissions|CO2 in 2050, Mt CO2/yr',
    'Emissions|Total Non-CO2 in 2050, Mt CO2-equiv/yr',
    'Carbon Sequestration|CCS in 2050, Mt CO2/yr',
    'Carbon Sequestration|CCS|Biomass in 2050, Mt CO2/yr',
    'Carbon Sequestration|CCS|Fossil in 2050, Mt CO2/yr',
    'Carbon Sequestration|CCS|Industrial Processes in 2050, Mt CO2/yr'
    ]

yticks = [v.split(' in 2050')[0] for v in varis]
yticks = [v.split('|')[1:] for v in yticks]
yticks = [x[0] if len(x)==1 else x[0]+' '+ x[1] for x in yticks]

plot_box_meta(dfb, varis, yticks, xlabel='Emissions and removals in 2050',
              fname=fname, palette=palette)


#%% Energy shares in 2050
fname = f'{output_folder}report_figures\\boxplot_energy_shares_2050.png'
varis = [
    'Primary Energy|Renewables (incl.Biomass)|Share in 2050, %',
    'Primary Energy|Non-biomass renewables|Share in 2050, %',
    'Primary Energy|Fossil|Share in 2050, %',
    'Primary Energy|Fossil|w/o CCS|Share in 2050, %',
    'Secondary Energy|Electricity|Renewables (incl.Biomass)|Share in 2050, %',
    'Secondary Energy|Electricity|Non-Biomass Renewables|Share in 2050, %',
    'Hydrogen production|Final Energy|Share in 2050, %',
    'Final Energy|Electrification|Share in 2050, %',
    'Final Energy|Industry|Fossil|Share in 2050, %',
    'Final Energy|Residential and Commercial|Fossil|Share in 2050, %',
    'Final Energy|Transportation|Fossil|Share in 2050, %',
    'PE Import dependency in 2050, %',
    'PE Import dependency|Fossil in 2050, %',
]
yticks = [v.split('|Share in 2050, %')[0] for v in varis]
yticks = [v.strip(' in 2050, %') for v in yticks]

plot_box_meta(dfb, varis, yticks, xlabel='% share in 2050',
              fname=fname, palette=palette)





