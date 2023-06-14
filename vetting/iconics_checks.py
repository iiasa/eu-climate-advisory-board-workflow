# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:05:41 2023

@author: byers
"""



import pyam
import pandas as pd

#%%


main_folder = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\'
vetting_output_folder = f'{main_folder}vetting\\'

vstr = '20230512'  
wbstr = f'{vetting_output_folder}vetting_flags_global_regional_combined_{vstr}_v4.xlsx'

vetting = pd.read_excel(wbstr, sheet_name='Vetting_flags')

instance = 'eu-climate-advisory-board-internal'

x = df.meta.reset_index()[['model','scenario']]
msdic = {k: list(v) for k,v in x.groupby("model")["scenario"]}


pyam.read_iiasa(instance,
                        model=model,
                        scenario=scenarios,
                        variable='Emissions|CO2',
                       region='World', meta=False),
            inplace=True)




#%%
user = 'byers'

main_folder = f'C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Projects\\EUAB\\'
vetting_output_folder = f'{main_folder}vetting\\'


vstr = '20230512'  
wbstr = f'{vetting_output_folder}vetting_flags_global_regional_combined_{vstr}_v4.xlsx'

data_output_folder = f'{main_folder}iconics\\{vstr}\\'




#%% CCS recalculate & checks
fn_out = f'{data_output_folder}iconics_NZ_data_and_table_{vstr}_v16v8.xlsx'

dfr = pyam.IamDataFrame(fn_out, sheet_name='data')

# For REMIND 2.1 / 3,2 scenarios
df = dfr.append(dfr.aggregate('CCUS', components=['Carbon Capture|Usage', 'Carbon Capture|Storage'],))

df.aggregate('CCS Miles (incl DAC)',['Carbon Sequestration|CCS', 'Carbon Sequestration|Direct Air Capture'],  append=True)


# others
dfo = df.filter(model=['REMIND 2.1', 'REMIND 3.2'], keep=False)
# dfo.add('Carbon Sequestration|CCS', 'Carbon Sequestration|Direct Air Capture','CCUS', ignore_units='Mt CO2/yr', append=True)
dfo.aggregate('CCUS',['Carbon Sequestration|CCS', 'Carbon Sequestration|Direct Air Capture'],  append=True)
df.append(dfo.filter(variable='CCUS'), inplace=True)



df.set_meta_from_data('Carbon Sequestration|CCS (orig) in 2050', variable='Carbon Sequestration|CCS', year=2050)
df.set_meta_from_data('CCS Miles (incl DAC) in 2050', variable='CCS Miles (incl DAC)', year=2050)
df.set_meta_from_data('CCUS in 2050', variable='CCUS', year=2050)
df.set_meta_from_data('CCUS in 2070', variable='CCUS', year=2070)



# REMIND 3.0 & WITHC 5.1 have Carbon Seq | DAC -0 ginroe because wil be excluded anyways

cols = ['cumulative GHGs** (incl. indirect AFOLU) (2030-2050, Gt CO2-equiv)',
             'GHG** emissions reductions 1990-2040 %',
             'Carbon Sequestration|CCS (orig) in 2050',
             'CCS Miles (incl DAC) in 2050',
             'CCUS in 2050',
             'diff_Miles_new',
             'CCUS in 2070', ]

feas_cols = [
            'Model','Scenario','feasibility',
            'primary_energy_biomass',
            'carbon_sequestration_ccs',
            'emissions_co2_lulucf_direct+indirect',
            'carbon_sequestration_direct_air_capture',
            'biomass_f',
            'ccs_f',]
                

feas_flags = pd.read_csv(f'{data_output_folder}filtering_v16v8_08_06_2023.csv')[feas_cols]
feas_flags.rename(columns={'Model':'model', 'Scenario':'scenario'}, inplace=True)
feas_flags.set_index(['model','scenario'], inplace=True)
dfm = df.meta
dfm['diff_Miles_new'] = dfm['CCUS in 2050'] - dfm['CCS Miles (incl DAC) in 2050']
dfm = pd.merge(dfm, feas_flags, left_index=True, right_index=True, how='outer')

tb = dfm.loc[dfm['Pass based on GHG** emissions reductions']==True, cols+feas_cols[2:]]
fn = f'{data_output_folder}ccus1.xlsx'
tb.to_excel(fn, merge_cells=False)              
os.startfile(fn)              






# =============================================================================
#%% Make comparison file
# =============================================================================
fn_out = f'{data_output_folder}iconics_NZ_data_and_table_{vstr}_v17.xlsx'
fn_out_prev = f'{data_output_folder}iconics_NZ_data_and_table_{vstr}_v16v8.xlsx'
fn_comparison = f'{data_output_folder}comparison_v16v5_v17_data.xlsx'

yrs = range(2019,2101)
old = pyam.IamDataFrame(fn_out_prev)
comparison = pyam.compare(old.filter(year=yrs), df.filter(year=yrs))
comparison.to_excel(fn_comparison, merge_cells=False)
os.startfile(fn_comparison)



# =============================================================================
#%% Check aggregation of variables
# =============================================================================

#%% Check 2100 diff

# (need) to load data first

# CO2 check
dfv = df.filter(region='EU27', unit='Mt CO2*', ghgfilter=True)
dfv.convert_unit( 'Mt CO2/yr', 'Mt CO2-equiv/yr',inplace=True)

dfv.multiply("Carbon Sequestration|Direct Air Capture", -1, 
             "Carbon Sequestration|Direct Air Capture-neg", 
             append=True, ignore_units='Mt CO2-equiv/yr')

comps = ['Emissions|CO2|Energy and Industrial Processes',
          "Emissions|CO2|LULUCF Direct+Indirect",
            "Carbon Sequestration|Direct Air Capture-neg"]
dfv.aggregate('Emissions|CO2|BU', components=comps, append=True)

dfv.subtract('Emissions|CO2',
             'Emissions|CO2|BU',
             name='Emissions|CO2-diff',
             append=True, ignore_units='Mt CO2-equiv/yr')

dfv.filter(variable='Emissions|CO2-diff').plot()
dfv.filter(variable='Emissions|CO2-diff').to_excel('c:\\users\\byers\\downloads\\co2diff.xlsx')
os.startfile('c:\\users\\byers\\downloads\\co2diff.xlsx')

# GHG checks
comps = ['Emissions|CO2|Energy and Industrial Processes',
          "Emissions|CO2|LULUCF Direct+Indirect",
            "Carbon Sequestration|Direct Air Capture-neg",
            'Emissions|Total Non-CO2']


dfv.aggregate('Emissions|Kyoto Gases (incl. indirect AFOLU)|BU', components=comps, append=True)
dfv.subtract('Emissions|Kyoto Gases (incl. indirect AFOLU)',
            'Emissions|Kyoto Gases (incl. indirect AFOLU)|BU', 
            name='Emissions|Kyoto Gases (incl. indirect AFOLU)-diff',
            ignore_units='Mt CO2-equiv/yr',
            append=True)


dfv.filter(variable=['Emissions|Kyoto Gases (incl. indirect AFOLU)',
            'Emissions|Kyoto Gases (incl. indirect AFOLU)|BU', 'Emissions|Kyoto Gases (incl. indirect AFOLU)-diff']).to_excel('c:\\users\\byers\\downloads\\ghg.xlsx')
os.startfile('c:\\users\\byers\\downloads\\ghg.xlsx')

dfv.filter(variable='Emissions|Kyoto Gases (incl. indirect AFOLU)').plot()
dfv.filter(variable='Emissions|Kyoto Gases (incl. indirect AFOLU)|BU').plot()
dfv.filter(variable='Emissions|Kyoto Gases (incl. indirect AFOLU)-diff').plot()


#%%
comps = ['Emissions|CO2|Energy and Industrial Processes',
          "Emissions|CO2|LULUCF Direct+Indirect",
            "Carbon Sequestration|Direct Air Capture-neg",
            'Emissions|Total Non-CO2']

dfv.filter(variable='Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)|BU', keep=False, inplace=True)
dfv.filter(variable='Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)-diff', keep=False, inplace=True)

dfv.aggregate('Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)|BU', components=comps, append=True)
dfv.subtract('Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)',
            'Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)|BU', 
            name='Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)-diff',
            ignore_units='Mt CO2-equiv/yr',
            append=True)

dfv.filter(variable='Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)-diff').plot()

#%%   intra minus no bunkers
name='Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)-nobunkers'
dfv.filter(variable=name, keep=False, inplace=True)
dfv.subtract('Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)',
            'Emissions|Kyoto Gases (incl. indirect AFOLU)', 
            name=name,
            ignore_units='Mt CO2-equiv/yr',
            append=True)

dfv.filter(variable=name).plot()

#%%   full ghg minus no bunkers
name='Emissions|Kyoto Gases (AR4) (EEA)-nobunkers'
dfv.filter(variable=name, keep=False, inplace=True)
dfv.subtract('Emissions|Kyoto Gases (AR4) (EEA)',
            'Emissions|Kyoto Gases (incl. indirect AFOLU)', 
            name=name,
            ignore_units='Mt CO2-equiv/yr',
            append=True)

dfv.filter(variable=name).plot()
