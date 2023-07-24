# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 18:10:31 2023
@author: byers

Execute this script from within the "vetting" folder

The aim of this script is to query the database, read data and indicators and 
produce the "iconics table" that was provided to EUABCC to facilitate assessment. 
For calculation of the additional variables and metadata indicators, see script 4_iconics_table_reprod_meta_indicators.py

"""

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
import matplotlib.pyplot as plt
import seaborn as sns
# os.chdir('C:\\Github\\eu-climate-advisory-board-workflow\\vetting')
from vetting_functions import *


user = 'byers'
output_folder = f'outputs\\'
instance = 'eu-climate-advisory-board'
datestr = '20230712'  
fn_out = f'{output_folder}iconics_NZ_data_and_table_{datestr}_v17.xlsx'


years = list(range(2015,2071))

prefix = 'Diagnostics|Harmonized|'

# These are the variables to read in from the database. (Not strictly necessary 
# to have all these..)
varlist = ['Diagnostics*',
           'Final Energy*',
           'Trade*',
           'Imports*',
           'Carbon Capture, Use, and Sequestration', 
           'Carbon Sequestration*',
           'Hydrogen production|Final Energy|Share',
           'Primary Energy|*Share',
           'Secondary Energy|Electricity|*Share',
           'PE Import dependency*',
           'Primary Energy|Import Dependency*',
           ]



#% load pyam data
dfin = pyam.read_iiasa(instance,
                       variable=varlist,
                       year=years,
                       region='EU27', meta=True)



# Keep only the trade data in 'EJ/yr'
dfin.filter(variable='Trade*', unit='billion US$2010/yr', 
            keep=False, inplace=True)

df = dfin.filter(region='EU27')

# Filter years in case of odd zeros
years = list(range(1990,2016,5))+[2019]+list(range(2020,2071,5))
df.filter(year=years, inplace=True)

df.filter(**{'Vetting status':'pass'}, inplace=True)

df.filter(Category=['C1*', 'Regional only'], inplace=True)



# Manual fix - replace meta with false values to 0
# https://github.com/byersiiasa/eu-climate-advisory-board-workflow/issues/4


columns_list =  ['Final Energy|Residential and Commercial|Fossil|Share in year of net zero, %',
                 'Final Energy|Residential and Commercial|Fossil|Share in 2050, %',
                 'Carbon Sequestration|CCS|Fossil in year of net zero, Mt CO2/yr',
                 'Carbon Sequestration|CCS|Fossil in 2050, Mt CO2/yr',
                          ]

rdic = {x:'false' for x in columns_list}
df.meta.replace(rdic, 0.0, inplace=True)             
             
             
#%% Write out

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

# Write out meta sheet
write_meta_sheet(df, fn_out, startfile=True)


# =============================================================================
#%% Make the boxplots in the report
# =============================================================================

df.meta['ghgfilter'] = df.meta[f'Pass based on GHG** emissions reductions']

dfb = df.meta
dfb = dfb.loc[dfb[f'Pass based on GHG** emissions reductions']=='true']

# Make folders if necessar
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



