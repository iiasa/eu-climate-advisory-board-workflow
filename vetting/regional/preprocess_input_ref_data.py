#%%
import country_converter as coco
cc = coco.CountryConverter()
import pandas as pd
import pyam

user = 'byers'
wdg = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\global\\input_data\\'
wdr = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\regional\\input_data\\'
wd =  f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\input_data\\'

edgar_year = range(1990,2020)
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
# Include ONLY relevant IPCC codes for Energy
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

dfch4 = dfch4.loc[dfch4.fossil_bio=='fossil']
# sum fossil and bio CH4 here, no need to separate
dfe4 = dfch4[['region', 'fossil_bio', 'year', 'value']].groupby(['region','year']).sum()

dfe4 = dfe4.reset_index()#.rename(columns={'ISO':'region'})
dfe4[['model','scenario','variable','unit']] = ['Reference','EDGAR AR6','Emissions|CH4','Mt CH4/yr']
dfe4['value'] = dfe4['value']/1e6 # convert from t to Mt
dfe4.value.sum()

dfedgarCH4 = pyam.IamDataFrame(dfe4)

#%% F-GASES

# dffg = pd.read_excel(f'{wd}pre_processing_data\\ipcc_ar6_data_edgar6_FGAS.xlsx',
#                     sheet_name='data', usecols=['ISO', 'fossil_bio', 'year','value'])

# dffg.rename(columns={'ISO':'region'}, inplace=True)


# # sum fossil and bio CH4 here, no need to separate
# dfefg = dffg[['region', 'fossil_bio', 'year', 'value']].groupby(['region','year']).sum()

# dfefg = dfefg.reset_index()#.rename(columns={'ISO':'region'})
# dfefg[['model','scenario','variable','unit']] = ['Reference','EDGAR AR6','Emissions|CH4','Mt CH4/yr']
# dfefg['value'] = dfefg['value']/1e6 # convert from t to Mt
# dfefg.value.sum()

# dfedgarFAGSES = pyam.IamDataFrame(dfefg)

#%% merge and write out EDGAR data

dfedgar = dfedgarCO2.append(dfedgarCH4)

#% Aggregate to world

for v in dfedgar.variable:
    dfedgar.aggregate_region(variable=v,
                             region='World',
                             append=True)
    
# Aggregate to EU27    
iso_eu27 = ['AUT', 'BEL', 'BGR', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HRV', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE', 'NLD']
for v in dfedgar.variable:
    dfedgar.aggregate_region(variable=v,
                             region='EU27',
                             subregions=iso_eu27,
                             append=True)
# Filter year
dfedgar.filter(year=edgar_year, inplace=True)

dfedgar.to_csv(f'{wd}input_reference_edgarCO2CH4.csv')

#%% =============================================================================
# EEA Inventory data
# =============================================================================

dfeeain = pd.read_csv(f'{wd}pre_processing_data\\UNFCCC_v25.csv',
                    usecols=['Country_code', 'Pollutant_name',
                             'Sector_code',	'Sector_name',
                             'Parent_sector_code',	'Unit',	'Year',	'emissions'],
                    encoding='utf-8')

dfeeain.rename(columns={'Country_code':'region',
                     'Pollutant_name': 'variable',
                     'Unit': 'unit',
                     'Year': 'year',
                     'emissions': 'value'                   
                     }, inplace=True)

ghg_var = 'Kyoto Gases (AR4)'
dfeea = dfeeain.replace({'variable': {'All greenhouse gases - (CO2 equivalent)': ghg_var}})

keep_vars = [ghg_var, 'CH4', 'CO2',]#  'HFCs - (CO2 equivalent)', 'N2O', 'NF3 - (CO2 equivalent)',  'PFCs - (CO2 equivalent)', 'SF6 - (CO2 equivalent)', 'Unspecified mix of HFCs and PFCs - (CO2 equivalent)']


keep_sectors = ['1 - Energy', '2 - Industrial Processes and Product Use',  '3 - Agriculture',
                '4 - Land Use, Land-Use Change and Forestry', '5.A - Solid Waste Disposal',
                '6 - Other Sector',
                'Total net emissions (UNFCCC)',
                'Total net emissions with international aviation (EU NDC)',
                'Total net emissions with international transport (EEA)',]
dfeea = dfeea.loc[(dfeea.Sector_name.isin(keep_sectors) & (dfeea.variable.isin(keep_vars)) )]

# Fix year data
dfeea = dfeea.loc[dfeea.year!='1985-1987']
dfeea['year'] = dfeea.year.astype(int)


#% chnage units

dfeea.loc[(dfeea.unit=='Gg') & (dfeea.variable=='CO2'), 'unit'] = 'kt CO2/yr'
dfeea.loc[(dfeea.unit=='Gg') & (dfeea.variable=='CH4'), 'unit'] = 'kt CH4/yr'
dfeea.loc[(dfeea.unit=='Gg CO2 equivalent') & (dfeea.variable==ghg_var), 'unit'] = 'kt CO2-equiv/yr'

gases = dfeea.variable.unique()
#%
for gas in gases:

    rename_dic = {
                    '1 - Energy': f'Emissions|{gas}|Energy',
                    '2 - Industrial Processes and Product Use': f'Emissions|{gas}|Industrial Processes',
                    '3 - Agriculture': f'Emissions|{gas}|Agriculture',
                    '4 - Land Use, Land-Use Change and Forestry': f'Emissions|{gas}|LULUCF',
                    '5.A - Solid Waste Disposal': f'Emissions|{gas}|Waste',
                    '6 - Other Sector': f'Emissions|{gas}|Other',
                    'Total net emissions (UNFCCC)': f'Emissions|{gas} (UNFCCC)',
                    'Total net emissions with international aviation (EU NDC)': f'Emissions|{gas} (EU NDC)',
                    'Total net emissions with international transport (EEA)': f'Emissions|{gas} (EEA)',
                    }

    dfeea.loc[dfeea.variable==gas, 'Sector_name'] = dfeea.loc[dfeea.variable==gas, 'Sector_name'].replace(rename_dic)

    # dfeea['Sector_name'] = .dfeea['Sector_name']replace(rename_dic, inplace=True)
dfeea['variable'] = dfeea['Sector_name']


dfeea['model'] = 'Reference'
dfeea['scenario'] = 'EEA GHG 2021'


#%
dfeeap = dfeea[pyam.IAMC_IDX+['year','value']]
dfeeap = pyam.IamDataFrame(dfeeap)
keep_years = range(1990,2023)
dfeeap.filter(year=keep_years, inplace=True)

#% Regions
regions = {x:cc.pandas_convert(series=pd.Series(x), to='ISO3', not_found=None)[0] for x in dfeeain.region.unique()} 
regions['EUA'] = 'EU27 & UK'
regions['EUC'] = 'EU-KP'
regions['EUX'] = 'EU27'

dfeeap.rename(mapping={'region': regions}, inplace=True)


# Units
dfeeap.convert_unit('kt CO2/yr', 'Mt CO2/yr',  inplace=True)
dfeeap.convert_unit('kt CH4/yr', 'Mt CH4/yr',  inplace=True)
dfeeap.convert_unit('kt CO2-equiv/yr', 'Mt CO2-equiv/yr',  inplace=True)

for gas in gases:
    v = f'Emissions|{gas}|Energy and Industrial Processes'
    components = [f'Emissions|{gas}|Energy', f'Emissions|{gas}|Industrial Processes', ]
    dfeeap.aggregate(v, components=components,
                     append=True)
    
    
for agg in ['EEA','EU NDC','UNFCCC']:
    v = f'Emissions|Total Non-CO2 (AR4) ({agg})'
    a = f'Emissions|{ghg_var} ({agg})'
    b = f'Emissions|CO2 ({agg})'
    dfeeap.subtract(a, b, v, append=True, ignore_units='Mt CO2-equiv/yr')
    
    
dfeeap.to_csv(f'{wd}input_reference_EEA.csv')


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
    
    dfirena.aggregate_region(variable=v,
                             region='EU27', subregions=iso_eu27, append=True)

dfirena.rename(mapping={'region': regions}, inplace=True)


dfirena.convert_unit('GWh/yr','EJ/yr', inplace=True)
dfirena.add('Secondary Energy|Electricity|Solar', 'Secondary Energy|Electricity|Wind', 'Secondary Energy|Electricity|Solar-Wind', append=True)

dfirena.rename({'region':{'EU 27':'EU27'}}, inplace=True)

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
dfemb.loc[dfemb.Country=='EU', 'region'] = 'EU27'

dfemb = dfemb.loc[dfemb.region.isna()==False]


subset_variables = ['Solar','Wind','Wind and Solar']
dfemb = dfemb.loc[dfemb.Variable.isin(subset_variables)]

repdic = {'Wind and Solar': 'Secondary Energy|Electricity|Solar-Wind',
          'Solar': 'Secondary Energy|Electricity|Solar',
          'Wind': 'Secondary Energy|Electricity|Wind'}

dfemb.replace(repdic, inplace=True)
dfemb[['model','scenario',]] = ['Reference', 'EMBERS GER 2022']


dfemb = pyam.IamDataFrame(dfemb.drop(columns='Country'))

# for v in dfemb.variable:
#     dfemb.aggregate_region(variable=v,
#                              region='EU27', subregions=iso_eu27, append=True)


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


# PE_spec = {'Nuclear': 0.3,
#            'Geothermal': 0.1,
#            # 'Solar|Thermal': 0.1,
#            }

# for tech, fac in PE_spec.items():

#     tv = dfiear.multiply(f'Primary Energy|{tech}', 
#                          fac, f'Primary Energy|{tech}',
#                          ignore_units='EJ/yr')
#     dfiear.filter(variable=f'Primary Energy|{tech}', keep=False, inplace=True)
#     dfiear.append(tv, inplace=True)


# # recalculate aggregate
dfiear.filter(variable=f'Primary Energy', keep=False, inplace=True)
components = [x for x in dfiear.filter(variable=f'Primary Energy|*').variable if 'Fossil' not in x]
dfiear.aggregate('Primary Energy', components=components, append=True)



#%% IEA World data

dfieaw = pyam.IamDataFrame(f'{wd}pre_processing_data\\IEA - Total energy supply (TES) and Elec Gen by source - World.xlsx', sheet_name='data')

dfiea = dfiear.append(dfieaw)

for v in dfiea.variable:
    print(v)
    dfiea.aggregate_region(variable=v,
                             region='EU27', subregions=iso_eu27, append=True)

dfiea.to_csv(f'{wd}input_reference_iea.csv')


# =============================================================================
#%% Merge all and make Solar-Wind composite
# =============================================================================

# Merge together
dfall = dfedgar.append(dfirena)
dfall.append(dfemb, inplace=True)
dfall.append(dfiea, inplace=True)  # Both regional and world data
dfall.append(dfeeap, inplace=True)

# make Solar-Wind composite
dfswc = dfall.filter(variable='Secondary Energy|Electricity|Solar-Wind')
dfswcp = dfswc.as_pandas(meta_cols=False)

dfswcp = dfswcp.groupby(['region','year']).mean().reset_index()
dfswcp[msvu] = ['Reference', 'Solar-Wind-composite', 'Secondary Energy|Electricity|Solar-Wind', 'EJ/yr']
dfswcp = pyam.IamDataFrame(dfswcp)

# dfswcp.aggregate_region(variable='Secondary Energy|Electricity|Solar-Wind',
#                              region='EU27',
#                              subregions=iso_eu27,
#                              append=True)


dfall.append(dfswcp, inplace=True)

# Save out

dfall.to_csv((f'{wd}input_reference_all.csv'))
