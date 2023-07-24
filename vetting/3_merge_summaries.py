# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:07:51 2023

@author: byers
"""
# Execute this script from within the "vetting" folder

# SUMMARY -= MERGE GLOBAL AND REGIONAL
import os
import pandas as pd
import pyam
# os.chdir('C:\\Github\\eu-climate-advisory-board-workflow\\vetting')
from vetting_functions import *

datestr = '20230712'
user = 'byers'

output_folder = f'outputs\\'

fnr = f'{output_folder}vetting_flags_all_regional_{datestr}.xlsx'
fng = f'{output_folder}vetting_flags_all_global_{datestr}.xlsx'
wbstr = f'{output_folder}vetting_flags_global_regional_combined_{datestr}_v4.xlsx'

dfr = pd.read_excel(fnr, sheet_name='vetting_flags')
dfg = pd.read_excel(fng, sheet_name='vetting_flags')

ms = ['model','scenario',]

colg = 'vetting_global'
colr = 'vetting_regional'

flag_pass = 'PASS'
flag_pass_missing = 'Pass_missing'
flag_fail_missing = 'Fail_missing'
flag_fail = 'FAIL'

colsg = ms+[colg, 
       'Key_historical', 'Key_future', flag_pass_missing, flag_fail_missing]
colsr = ms+[colr, 
       'Key_historical', 'Key_future', flag_pass_missing, flag_fail_missing]
    

dfs = pd.merge(dfg[colsg], dfr[colsr+['vetted_type',]], 
               on=ms, 
               how='outer',
               suffixes=('_G','_R'))


#%% Recode the values with Identifiers

col = 'OVERALL_binary'
col1 = 'OVERALL_Assessment'
col2 = 'OVERALL_rank'


# 1. Double PASS
dfs.loc[(dfs[colg]==flag_pass) & (dfs[colr]==flag_pass), col] = flag_pass
dfs.loc[(dfs[colg]==flag_pass) & (dfs[colr]==flag_pass), col1] = flag_pass
dfs.loc[(dfs[colg]==flag_pass) & (dfs[colr]==flag_pass), col2] = 1

# 2. Pass plus a Pass_missing                                  
dfs.loc[(dfs[colg]==flag_pass) & (dfs[colr]==flag_pass_missing), col] = flag_pass
dfs.loc[(dfs[colg]==flag_pass) & (dfs[colr]==flag_pass_missing), col1] = flag_pass_missing
dfs.loc[(dfs[colg]==flag_pass) & (dfs[colr]==flag_pass_missing), col2] = 2

dfs.loc[(dfs[colg]==flag_pass_missing) & (dfs[colr]==flag_pass), col] = flag_pass
dfs.loc[(dfs[colg]==flag_pass_missing) & (dfs[colr]==flag_pass), col1] = flag_pass_missing
dfs.loc[(dfs[colg]==flag_pass_missing) & (dfs[colr]==flag_pass), col2] = 2

# 3. Double Pass_missing or NA_G+PASS_R
dfs.loc[(dfs[colg]==flag_pass_missing) & (dfs[colr]==flag_pass_missing), col] = flag_pass
dfs.loc[(dfs[colg]==flag_pass_missing) & (dfs[colr]==flag_pass_missing), col1] = 'Double_missing'
dfs.loc[(dfs[colg]==flag_pass_missing) & (dfs[colr]==flag_pass_missing), col2] = 3

dfs.loc[(dfs[colg].isna()) & (dfs[colr]==flag_pass), col] = flag_pass
dfs.loc[(dfs[colg].isna()) & (dfs[colr]==flag_pass), col1] = 'Regional only'
dfs.loc[(dfs[colg].isna()) & (dfs[colr]==flag_pass), col2] = 3

# 4. Fail Global, regional with warnings
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass), col] = flag_fail
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass), col1] = 'Global fail, Regional Good'
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass), col2] = 4

dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass_missing), col] = flag_fail
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass_missing), col1] = 'Global fail, Regional OK'
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass_missing), col2] = 4

# 4. NA Global, missing variables in regional
dfs.loc[(dfs[colg].isna()) & (dfs[colr]==flag_pass_missing), col] = 'WARNING'
dfs.loc[(dfs[colg].isna()) & (dfs[colr]==flag_pass_missing), col1] = 'Regional only OK'
dfs.loc[(dfs[colg].isna()) & (dfs[colr]==flag_pass_missing), col2] = 4


# Fail_missing_G, Pass_missing_G
dfs.loc[(dfs[colg]==flag_fail_missing) & (dfs[colr]==flag_pass_missing), col] = flag_fail
dfs.loc[(dfs[colg]==flag_fail_missing) & (dfs[colr]==flag_pass_missing), col1] = 'Global fail, Regional OK'
dfs.loc[(dfs[colg]==flag_fail_missing) & (dfs[colr]==flag_pass_missing), col2] = 4

# Regional Fail

# if Regional is FAIL or NAN, set to FAIL
dfs.loc[dfs[colr]==flag_fail, col] = flag_fail
dfs.loc[dfs[colr]==flag_fail, col1] = 'Regional fail'
dfs.loc[dfs[colr]==flag_fail, col2] = 0

dfs.loc[dfs[colr].isna(), col] = flag_fail
dfs.loc[dfs[colr].isna(), col1] = 'No Regional'
dfs.loc[dfs[colr].isna(), col2] = 0

# Not necessary - but keep just in case
# if Regional is PASS, set to PASS, unless Global fail
# dfs.loc[dfs[colr]==flag_pass, col] = flag_pass
# dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass), col] = 'FailG - PassR'
# dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass_missing), col] = 'FailG - MissingR'

# dfs.loc[(dfs[colg]==flag_pass_missing) & (dfs[colr]==flag_pass_missing), col] = flag_missing
# dfs.loc[(dfs[colg]=='PASS') & (dfs[colr]=='PASS'), col] = flag_


dfs[colg].fillna('NA', inplace=True)
dfs[colr].fillna('NA', inplace=True)

dfs[colg+'_G'] = dfs[colg]+'_G'
dfs[colr+'_R'] = dfs[colr]+'_R'

col = 'OVERALL_code'

dfs[col] = dfs[colg+'_G']+'+'+dfs[colr+'_R']


#%% Check for unclassified rows

ocols = [col for col in dfs.columns if 'OVERALL' in col]

if sum(dfs[ocols].isna().sum())>0:
    print('Warning - some rows not classified')
    sdfsdfsdf # break
    
    
# Drop REMIND 3.2 scenarios (in case still here)
dfs = dfs.loc[~((dfs.model=='REMIND 3.2') & (dfs.scenario.str.contains('withICEPhOP')==False))]

# Drop Reference
dfs = dfs.loc[~(dfs.model=='Reference')]


#%% Add climate metadata
# First pass
c_cols = ['Category', 'Category_name']
keep_cols = ['model','scenario']+c_cols

cmd1 = pd.read_excel(f'input_data\\EU_CAB_World_Emissions_meta.xlsx')
cmd1.loc[cmd1.scenario=='NGFS-Below 2°C', 'scenario'] = 'NGFS-Below 2C'
cmd1.loc[cmd1.scenario=='NGFS-Delayed transition', 'scenario'] = 'NGFS-Delayed Transition'

cmd4 = pd.read_excel(f'input_data\\EU_CAB_World_Emissions_meta_extra.xlsx') # Contains the two above.
cmd = pd.concat([cmd1,  cmd4])
cmd.drop_duplicates(subset=['model','scenario'], inplace=True)

cmd = cmd.loc[~((cmd.model=='REMIND 3.2') & (cmd.scenario.str.contains('withICEPhOP')==False))]

dfs = dfs.merge(cmd[keep_cols], on=['model','scenario'], how='outer')

# Drop DIAG-NZero scenarios (removed)
# # REMIND
dfs = dfs.loc[~(dfs.scenario=='DIAG-NZero')]

#%% Check for duplicates


# Add selected climate meta to main summary sheet
# =============================================================================
# Wrtite out EXCEL
# =============================================================================

writer = pd.ExcelWriter(wbstr, engine='xlsxwriter')

# Main summary sheet
cols = [x for x in dfs.columns if x not in ocols]
colorder = ms+ocols+c_cols+cols[2:-2]
dfs = dfs[colorder]
dfs.to_excel(writer, sheet_name='Vetting_flags', index=False, header=True)


# FORMATTING
workbook  = writer.book

 # vetting_flags page
worksheet = writer.sheets['Vetting_flags']
header_format = workbook.add_format(
    {'bold': True, 
     'text_wrap': True, 
     'valign': 'top'})
    
worksheet.set_column(0, 0, 15, None)
worksheet.set_column(1, 1, 25, None)
worksheet.set_column(2, len(dfs.columns)-1, 20, None)
worksheet.freeze_panes(1, 3)
worksheet.autofilter(0, 0, len(dfs), len(dfs.columns)-1)



# Pivot table sheets
dfsp = simple_pivot_cat_count(dfs[['OVERALL_Assessment','OVERALL_binary']],
                                columns='OVERALL_Assessment', 
                                index='OVERALL_binary', )
write_simple_pivot(writer, dfsp, sheet_name='Pivot_Assessment')


dfsp = simple_pivot_cat_count(dfs[['OVERALL_code','OVERALL_binary']],
                                columns='OVERALL_code', 
                                index='OVERALL_binary', )
write_simple_pivot(writer, dfsp.T, sheet_name='Pivot_code')

dfsp = simple_pivot_cat_count(dfs[['vetted_type','OVERALL_binary']],
                                columns='vetted_type', 
                                index='OVERALL_binary', )
write_simple_pivot(writer, dfsp, sheet_name='Pivot_vetted_type')
    


# Write out climate metadata
cmd.to_excel(writer, sheet_name='meta_climate', index=False)

writer.close()

os.startfile(wbstr)