#%%
import country_converter as coco
cc = coco.CountryConverter()
import pandas as pd
import pyam

user = 'byers'
wdg = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\global\\input_data\\'
wdr = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\regional\\input_data\\'
wd =  f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\input_data\\'

edgar_year = range(2000,2020)
iea_year = range(2000,2020)

msvu = ['model','scenario','variable','unit']


#%% =============================================================================
# EDGAR
# =============================================================================
#%% Import and subset the EDGAR CO2 data
dfco2all = pd.read_excel(f'{wd}pre_processing_data\\ipcc_ar6_data_edgar6_CO2.xlsx',
                    sheet_name='data', usecols=['ISO','year', 'chapter_title',
                                                'sector_code','value'])

dfco2all.rename(columns={'ISO':'region'}, inplace=True)


# Full economy CO2 (not used, because LULUCF only reported for R10 regions)
# dfco2 = dfco2all[['region','year','value']].groupby(['region','year']).sum()
# dfco2 = dfco2.reset_index()#.rename(columns={'ISO':'region'})
# dfco2[msvu] = ['Reference','EDGAR AR6','Emissions|CO2','Mt CO2/yr']
# dfco2['value'] = dfco2['value']/1e6 # convert from t to Mt


# Location of LULCF CO2 data
# P:/ene.general/ar6snap/edgar/EDGAR_FGD_FINAL/ipcc_ar6_data_edgar6_LULUCF.xlsx


# dfedgarCO2 = pyam.IamDataFrame(dfco2)

#%% Energy & Industrial processes
dfeip = dfco2all.loc[dfco2all.chapter_title!='AFOLU']
dfeip = dfeip[['region','year','value']].groupby(['region','year']).sum()
dfeip = dfeip.reset_index()#.rename(columns={'ISO':'region'})
dfeip[msvu] = ['Reference','EDGAR AR6','Emissions|CO2|Energy and Industrial Processes','Mt CO2/yr']
dfeip['value'] = dfeip['value']/1e6 # convert from t to Mt
dfeip.value.sum()

# Energy
dfe = dfco2all.loc[dfco2all.chapter_title.isin(['AFOLU'])==False]
# Include ONLY relevant IPCC codes
ipcc_keep_codes = ('1A','1B')

dfe = dfe.loc[dfe.sector_code.str.startswith(ipcc_keep_codes)]


dfe = dfe[['region','year','value']].groupby(['region','year']).sum()
dfe = dfe.reset_index()#.rename(columns={'ISO':'region'})
dfe[msvu] = ['Reference','EDGAR AR6','Emissions|CO2|Energy','Mt CO2/yr']
dfe['value'] = dfe['value']/1e6 # convert from t to Mt
dfe.value.sum()

dfeip = pyam.IamDataFrame(dfeip)
dfe = pyam.IamDataFrame(dfe)

# dfedgarCO2 = dfedgarCO2.append(dfeip)
dfedgarCO2 = dfeip.append(dfe)

#%% Import and subset the EDGAR CH4 data

# dfi = pyam.IamDataFrame(f'{wd}pre_processing\\ipcc_ar6_data_edgar6_CH4.xlsx')
dfch4 = pd.read_excel(f'{wd}pre_processing_data\\ipcc_ar6_data_edgar6_CH4.xlsx',
                    sheet_name='data', usecols=['ISO', 'fossil_bio', 'year','value'])

dfch4.rename(columns={'ISO':'region'}, inplace=True)


# sum fossil and bio CH4 here, no need to separate
dfe4 = dfch4[['region', 'fossil_bio', 'year', 'value']].groupby(['region','year']).sum()

dfe4 = dfe4.reset_index()#.rename(columns={'ISO':'region'})
dfe4[['model','scenario','variable','unit']] = ['Reference','EDGAR AR6','Emissions|CH4','Mt CH4/yr']
dfe4['value'] = dfe4['value']/1e6 # convert from t to Mt
dfe4.value.sum()

dfedgarCH4 = pyam.IamDataFrame(dfe4)



#%% merge and write out EDGAR data

dfedgar = dfedgarCO2.append(dfedgarCH4)

#% Aggregate to world

for v in dfedgar.variable:
    dfedgar.aggregate_region(variable=v,
                             region='World',
                             append=True)

# Filter year
dfedgar.filter(year=edgar_year, inplace=True)

dfedgar.to_csv(f'{wd}input_reference_edgarCO2CH4.csv')


#%% =============================================================================
# IRENA
# =============================================================================

#%% Irena wind-solar by region

dfirena = pd.read_excel(f'{wd}pre_processing_data\\IRENA_ref_data.xlsx', sheet_name='IRENA_all')
# dfirena.drop(columns=['Technology', 'region'], inplace=True)
# dfirena.rename({'ISO','region'}, axis=1, inplace=True)
dfirena = pyam.IamDataFrame(dfirena.drop(columns=['Technology']))

regions = {x:cc.pandas_convert(series=pd.Series(x), to='ISO3', not_found=None)[0] for x in dfirena.region} 

# Aggregate odd china regions
chns = [k for k,v in regions.items() if v=='CHN']

for v in dfirena.variable:
    dfirena.aggregate_region(variable=v,
                             region='CHN', subregions=chns, append=True)
    dfirena.filter(variable=v, region=chns, keep=False, inplace=True)

dfirena.rename(mapping={'region': regions}, inplace=True)


dfirena.convert_unit('GWh/yr','EJ/yr', inplace=True)
dfirena.add('Secondary Energy|Electricity|Solar', 'Secondary Energy|Electricity|Wind', 'Secondary Energy|Electricity|Solar-Wind', append=True)


dfirena.to_csv(f'{wd}input_reference_irena.csv')


#%% =============================================================================
# EMBERS
# =============================================================================

#%% EMBERS 
dfemb = pd.read_excel(f'{wd}pre_processing_data\\Ember-GER-2022-Data.xlsx',
                    sheet_name='Generation', usecols=['Country or region','Country code', 'Year','Variable','Electricity generated (TWh)'])

dfemb.rename(columns={'Country or region': 'Country',
                     'Country code': 'region',
                     'Electricity generated (TWh)': 'value'},
            inplace=True)

dfemb['value'] = dfemb['value']*0.0036  # Convert from TWh to EJ/yr
dfemb['unit'] = 'EJ/yr'

dfemb.loc[dfemb.Country=='World', 'region'] = 'World'
dfemb = dfemb.loc[dfemb.region.isna()==False]


subset_variables = ['Solar','Wind','Wind and Solar']
dfemb = dfemb.loc[dfemb.Variable.isin(subset_variables)]

repdic = {'Wind and Solar': 'Secondary Energy|Electricity|Solar-Wind',
          'Solar': 'Secondary Energy|Electricity|Solar',
          'Wind': 'Secondary Energy|Electricity|Wind'}

dfemb.replace(repdic, inplace=True)
dfemb[['model','scenario',]] = ['Reference', 'EMBERS GER 2022']



dfemb = pyam.IamDataFrame(dfemb.drop(columns='Country'))

dfemb.to_csv(f'{wd}input_reference_embers.csv')


#%% =============================================================================
# IEA
# =============================================================================
#%% Import and subset the IEA data

# IEA ISO data
dfiear = pyam.IamDataFrame(f'{wd}pre_processing_data\\IEA_history_IPCCSR15-ISO-2021-EB.xlsx',
                        sheet_name='all')

dfiear.filter(year=iea_year, inplace=True)
dfiear.rename({'model': {'History':'Reference'}}, inplace=True)

dfiear.add('Secondary Energy|Electricity|Solar', 'Secondary Energy|Electricity|Wind', 'Secondary Energy|Electricity|Solar-Wind', append=True)

#%% IEA World data

dfieaw = pyam.IamDataFrame(f'{wd}pre_processing_data\\IEA - Total energy supply (TES) and Elec Gen by source - World.xlsx', sheet_name='data')

dfiea = dfiear.append(dfieaw)

dfiea.to_csv(f'{wd}input_reference_iea.csv')


# =============================================================================
#%% Merge all and make Solar-Wind composite
# =============================================================================

# Merge together
dfall = dfedgar.append(dfirena)
dfall.append(dfemb, inplace=True)
dfall.append(dfiea, inplace=True)

# make Solar-Wind composite
dfswc = dfall.filter(variable='Secondary Energy|Electricity|Solar-Wind')
dfswcp = dfswc.as_pandas(meta_cols=False)

dfswcp = dfswcp.groupby(['region','year']).mean().reset_index()
dfswcp[msvu] = ['Reference', 'Solar-Wind-composite', 'Secondary Energy|Electricity|Solar-Wind', 'EJ/yr']

dfswcp = pyam.IamDataFrame(dfswcp)

# dfall.filter(variable='Secondary Energy|Electricity|Solar-Wind',
#              keep=False,
#              inplace=True)

dfall.append(dfswcp, inplace=True)


# Save out

dfall.to_csv((f'{wd}input_reference_all.csv'))


