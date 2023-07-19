# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 18:10:31 2023
@author: byers

# Execute this script from within the "vetting" folder

# The aim of this script is to query the database and produce the "iconics table"
# that was provided to EUABCC to facilitate assessment.

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

# wbstr = f'{output_folder}vetting_flags_global_regional_combined_{datestr}_v4.xlsx'

data_output_folder = output_folder #f'{main_folder}iconics\\{datestr}\\'

fn_out = f'{data_output_folder}iconics_NZ_data_and_table_{datestr}_v17.xlsx'


years = list(range(2015,2071))

prefix = 'Diagnostics|Harmonized|'

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
                        # model=models,
                        # scenario=scenarios,
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

sdfsdfsdf


#%% Which columns / order


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


# df_orig = pyam.IamDataFrame('C:/Users/byers/IIASA/ECE.prog - Documents/Projects/EUAB/iconics/20230512/iconics_NZ_data_and_table_20230512_v17.xlsx')
# df_orig = df_orig.filter(variable=varlist, year=years)
# dfoc = list(df_orig.meta.columns)


df.meta = df.meta[all_cols]

# dfmc = list(df.meta.columns)

# cols_both = list(set(dfoc).intersection(dfmc))

# # for col in dfoc




# for c in cols_both:
#     dt = df.meta[c].dtype
#     df_orig.meta[c] = df_orig.meta[c].astype(dt)

# compare = df_orig.meta[cols_both].compare(df.meta[cols_both])
# compare = df_orig.meta[cols_both].compare(df.meta[cols_both], keep_shape=False, keep_equal=False)
# compare = df_orig.meta[cols_both].compare(df.meta[cols_both], keep_shape=True, keep_equal=True)
# compare = df_orig.meta[cols_both].compare(df.meta[cols_both], keep_shape=True, keep_equal=False)
# df_orig.meta[cols_both].dtypes
# df.meta[cols_both].dtypes


# compare = df_orig.meta[cols_both] - df.meta[cols_both]

#%% Write out data sheet with formatting

writer = pd.ExcelWriter(fn_out, engine='xlsxwriter')

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

os.startfile(fn_out)

# df.meta.to_excel(fn_out.replace('data_and_',''),
                 # sheet_name='meta')


#%%
# metacols = ['Carbon Sequestration|CCS|Fossil in year of net zero, Mt CO2/yr', 'Carbon Sequestration|CCS|Fossil in 2050, Mt CO2/yr']
# df1 =  pyam.read_iiasa(instance,
#                         model='REMIND-MAgPIE 2.1-4.2',
#                         scenario=[
#                                     'EN_NPi2020_200f',
#                                     'EN_NPi2020_300f',
#                                     'EN_NPi2020_400',
#                                     'EN_NPi2020_400f',
#                                     'EN_NPi2020_500',
#                                     'EN_NPi2020_600'
#                                 ],
#                         variable='Diagnostics|Harmonized|Emissions|CO2',
#                        region='EU27', meta=True)


# df1.meta[metacols]

# #%

# metacols = ['Final Energy|Residential and Commercial|Fossil|Share in year of net zero, %', 'Final Energy|Residential and Commercial|Fossil|Share in 2050, %']
# df2 =  pyam.read_iiasa(instance,
#                         model='REMIND 3.2',
#                         scenario=[
#                                     'def_300_withICEPhOP',
#                                     'def_500_withICEPhOP',
#                                     'def_800_withICEPhOP',
#                                     'def_bioLim12_300_withICEPhOP',
#                                     'def_bioLim12_500_withICEPhOP',
#                                     'def_bioLim12_800_withICEPhOP',
#                                     'def_bioLim7p5_300_withICEPhOP',
#                                     'def_bioLim7p5_500_withICEPhOP',
#                                     'def_bioLim7p5_800_withICEPhOP',
#                                     'flex_300_withICEPhOP',
#                                     'flex_500_withICEPhOP',
#                                     'flex_800_withICEPhOP',
#                                     'flex_bioLim12_300_withICEPhOP',
#                                     'flex_bioLim12_500_withICEPhOP',
#                                     'flex_bioLim12_800_withICEPhOP',
#                                     'flex_bioLim7p5_300_withICEPhOP',
#                                     'flex_bioLim7p5_500_withICEPhOP',
#                                     'flex_bioLim7p5_800_withICEPhOP',
#                                     'NZero_bioLim12_withICEPhOP',
#                                     'NZero_bioLim7p5_withICEPhOP',
#                                     'NZero_withICEPhOP',
#                                     'rigid_300_withICEPhOP',
#                                     'rigid_500_withICEPhOP',
#                                     'rigid_800_withICEPhOP',
#                                     'rigid_bioLim12_300_withICEPhOP',
#                                     'rigid_bioLim12_500_withICEPhOP',
#                                     'rigid_bioLim12_800_withICEPhOP',
#                                     'rigid_bioLim7p5_300_withICEPhOP',
#                                     'rigid_bioLim7p5_500_withICEPhOP',
#                                     'rigid_bioLim7p5_800_withICEPhOP'
#                                 ],
#                         variable='Diagnostics|Harmonized|Emissions|CO2',
#                        region='EU27', meta=True)

# df2.meta[metacols]
