#%%
import pandas as pd
import pyam

user = 'byers'
wd = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\vetting\\regional\\input_data\\'

edgar_year = 2019
iea_year = 2020
#%% Import and subset the EDGAR CO2 data
df = pd.read_excel(f'{wd}pre_processing\\ipcc_ar6_data_edgar6_CO2.xlsx',
                    sheet_name='data', usecols=['ISO','year','value'])
dfe = df.loc[df.year==edgar_year]
dfe = dfe[['ISO','year','value']].groupby(['ISO','year']).sum()

dfe = dfe.reset_index().rename(columns={'ISO':'region'})
dfe[['model','scenario','variable','unit']] = ['Reference','EDGAR AR6','Emissions|CO2|Energy and Industrial Processes','Mt CO2/yr']
dfe['value'] = dfe['value']/1e6 # convert from t to Mt
dfe.value.sum()

dfe = pyam.IamDataFrame(dfe)

#%% Import and subset the EDGAR CH4 data

# dfi = pyam.IamDataFrame(f'{wd}pre_processing\\ipcc_ar6_data_edgar6_CH4.xlsx')
df = pd.read_excel(f'{wd}pre_processing\\ipcc_ar6_data_edgar6_CH4.xlsx',
                    sheet_name='data', usecols=['ISO', 'fossil_bio', 'year','value'])
dfe4 = df.loc[df.year==edgar_year]
# sum fossil and bio CH4 here, no need to separate
dfe4 = dfe4[['ISO', 'fossil_bio', 'year', 'value']].groupby(['ISO','year']).sum()


dfe4 = dfe4.reset_index().rename(columns={'ISO':'region'})
dfe4[['model','scenario','variable','unit']] = ['Reference','EDGAR AR6','Emissions|CH4','Mt CH4/yr']
dfe4['value'] = dfe4['value']/1e6 # convert from t to Mt
dfe4.value.sum()

dfe4 = pyam.IamDataFrame(dfe4)


#%% merge and write out EDGAR data

dfmerge = dfe.append(dfe4)

dfmerge.to_csv(f'{wd}input_reference_edgarCO2CH4.csv')

#%% Import and subset the IEA data

dfi = pyam.IamDataFrame(f'{wd}pre_processing\\Copy of IEAdb_extract_into_IAMC_format.xlsx',
                        sheet_name='data1')

dfi.filter(year=iea_year, inplace=True)
dfi.rename({'model': {'History':'Reference'}}, inplace=True)

dfi.to_csv(f'{wd}input_reference_ieaPE_SE.csv')

dfi.append(dfmerge).to_csv((f'{wd}input_reference_all.csv'))
