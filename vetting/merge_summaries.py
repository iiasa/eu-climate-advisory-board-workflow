# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:07:51 2023

@author: byers
"""

# SUMMARY -= MERGE GLOBAL AND REGIONAL

import pandas as pd
import pyam

user = 'byers'

output_folder = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\'

output_folderr = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\regional\\output_data\\'
output_folderg = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\global\\output_data\\'


fnr = f'{output_folderr}vetting_flags_all_regional.xlsx'
fng = f'{output_folderg}vetting_flags_all_global.xlsx'


dfr = pd.read_excel(fnr, sheet_name='vetting_flags')
dfg = pd.read_excel(fng, sheet_name='vetting_flags')

#%%
ms = ['model','scenario',]

colg = 'vetting_global'
colr = 'vetting_regional'

flag_pass = 'PASS'
flag_pass_missing = 'Pass_missing'
flag_fail = 'FAIL'

colsg = ms+[colg, 
       'Key_historical', 'Key_future', 'missing_count',]
colsr = ms+[colr, 
       'Key_historical', 'Key_future', 'missing_count',]    
    

dfs = pd.merge(dfg[colsg], dfr[colsr+['vetted_type',]], 
               on=ms, 
               how='outer',
               suffixes=('_G','_R'))



#%% Recode the values with Identifiers


#%%

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
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass), col] = 'WARNING'
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass), col1] = 'Regional only'
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass), col2] = 4

dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass_missing), col] = 'WARNING'
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass_missing), col1] = 'Regional only'
dfs.loc[(dfs[colg]==flag_fail) & (dfs[colr]==flag_pass_missing), col2] = 4

dfs.loc[(dfs[colg].isna()) & (dfs[colr]==flag_pass_missing), col] = 'WARNING'
dfs.loc[(dfs[colg].isna()) & (dfs[colr]==flag_pass_missing), col1] = 'Regional only'
dfs.loc[(dfs[colg].isna()) & (dfs[colr]==flag_pass_missing), col2] = 4

# Regional Fail

# if Regional is FAIL or NAN, set to FAIL
dfs.loc[dfs[colr]==flag_fail, col] = flag_fail
dfs.loc[dfs[colr]==flag_fail, col1] = 'Regional fail'
dfs.loc[dfs[colr]==flag_fail, col2] = 0

dfs.loc[dfs[colr].isna(), col] = flag_fail
dfs.loc[dfs[colr].isna(), col1] = 'No Regional'
dfs.loc[dfs[colr].isna(), col2] = 0

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


wbstr = f'{output_folder}vetting_flags_global_regional_combined.xlsx'
writer = pd.ExcelWriter(wbstr, engine='xlsxwriter')

cols = [x for x in dfs.columns if x not in ocols]
colorder = ms+ocols+cols[2:]

dfs = dfs[colorder]

dfs.to_excel(writer, sheet_name='summary', index=False, header=True)

 # vetting_flags page
 worksheet = writer.sheets['summary']
 worksheet.set_column(0, 0, 13, None)
 worksheet.set_column(1, 1, 25, None)
 worksheet.set_column(2, len(dfo.columns)-1, 20, None)
 worksheet.freeze_panes(1, 3)

