# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:01:43 2023
@author: byers, van der wijst

A collection of functions used for the vetting and auxiliary functions to pyam

"""

# vetting_functions.py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyam
import seaborn as sns
import string


#%% Define functions used to perform checks
log = True
print_log = print if log else lambda x: None
meta_docs = {}
historical_columns = []
future_columns = []
value_columns = {}

print_log = print if log else lambda x: None
###################################

# =============================================================================
#%%  VETTING
# Define functions used to perform vetting checks
# =============================================================================

def create_reference_df(df, ref_model, ref_scenario, check_variables, ref_year):
    check_variables = list(check_variables)
    return (
        df
        .filter(model=ref_model, scenario=ref_scenario, variable=check_variables, year=ref_year)
        .timeseries()
        [ref_year]
    )


## Some utilities for the filter functions:
#  - Initialise the meta column
#  - Perform check if a variable exists
#  - Set corresponding scenarios to `fail` given an output of *.validate() or *.check_aggregate

def util_filter_init(df, meta_name, meta_doc, meta_docs, key_historical, 
                     key_future, historical_columns, future_columns, flag_pass='Pass'):
    """Creates the meta column"""
    meta_docs[meta_name] = meta_doc
    df.set_meta(flag_pass, name=meta_name)
    df.reset_exclude()
    if key_historical==True:
        historical_columns.append(meta_name)
    if key_future==True:
        future_columns.append(meta_name)

def util_filter_check_exists(df, meta_name, variable, year=None, label='missing'):
    var_doesnt_exist = pyam.require_variable(df, variable=variable, year=year, exclude_on_fail=True)
    if var_doesnt_exist is not None:
        missing_indices = var_doesnt_exist[['model', 'scenario']]
        df.set_meta(label, meta_name, missing_indices)

def util_filter_set_failed(df, meta_name, failed_df, label='Fail'):
    print_log = print if log else lambda x: None

    if failed_df is None:
        print_log(f'All scenarios passed validation {meta_name}')
    else:
        print_log(f'{len(failed_df)} scenarios did not pass validation of {meta_name}')

        # depending on pyam version, if failed_df is a series (new pyam >=0.7),
        if type(failed_df)==pd.Series:
            failed_indices = (
                failed_df
                # .drop('variable',)
                .reset_index()
                [['model', 'scenario']]
                .drop_duplicates()
            )
        else: #or a DF (older pyam <0.7)
            failed_indices = (
                failed_df
                .drop('variable', axis=1)
                .reset_index()
                [['model', 'scenario']]
                .drop_duplicates()
            )


        df.set_meta(label, meta_name, failed_indices)
        df.set_meta(True, 'exclude', failed_indices)

def util_get_unit(df, variable):
    variables = df.as_pandas()[['variable','unit']].drop_duplicates()
    unit = variables[variables['variable'] == variable]['unit']
    try:
        return unit.iloc[0]
    except:
        return 'Unit not found'

def util_filter_save_value_column(df, name, variable_or_values, value_columns, year=None, method=None, ref_value=None, ref_lo=None, ref_up=None, plot=True, **kwargs):
    if type(variable_or_values) == str:
        unit = util_get_unit(df, variable_or_values)
        column_name = f'{name} (model value in {year}) [{unit}]'
        df.set_meta_from_data(name=column_name, variable=variable_or_values, year=year, method=method, **kwargs)
    else:
        unit = util_get_unit(df, variable_or_values.reset_index()['variable'].unique()[0])
        column_name = f'{name}' # (value)
        df.set_meta(variable_or_values, name=name, **kwargs)
    value_columns[column_name] = {
        'unit': unit,
        'ref_value': ref_value,
        'ref_lo': ref_lo,
        'ref_up': ref_up,
        'plot': plot
    }

## The filter functions:
#  - Filter with reference: compare to a reference value
#  - Filter check aggregate: check if components of variable sum up to main component
#  - Filter validate: check if variable is within bounds

def filter_with_reference(df, ref_df, thresholds, missing_flags, year, meta_name, meta_docs, key_historical=True, key_future=False, historical_columns=[], future_columns=[], value_columns={}, flag_fail='Fail', flag_pass='Pass',  ver='normal'):
    flag_pass_missing = flag_pass+'_missing'
    flag_fail_missing = flag_fail+'_missing'

    meta_name_agg = f'{meta_name} summary'
    trs = '' if ver=='teams' else f': {thresholds}'
    util_filter_init(df, meta_name_agg, f'Checks that scenario is within reasonable range'+trs,
                     meta_docs, key_historical, key_future, historical_columns, future_columns)

    curr_meta_columns = []
    # Loop over each variable and value
    for idx, val in ref_df.iteritems():
        _, _, curr_region, curr_variable, curr_unit, *_ = idx

        valstr = np.round(val, 2)
        curr_meta_column = f'({meta_name} {year}: {curr_variable})'#' = {valstr})'
        curr_meta_columns.append(curr_meta_column)
        df.set_meta(flag_pass, curr_meta_column)

        # Step 1: Identify scenarios where this variable/year is missing
        util_filter_check_exists(df, curr_meta_column, curr_variable, year, label=missing_flags[curr_variable])

        # Step 2: Identify scenarios where the value is out of range
        lo = val * (1-thresholds[curr_variable])
        up = val * (1+thresholds[curr_variable])
        util_filter_save_value_column(df, 
            curr_meta_column, curr_variable, value_columns=value_columns, year=year,
            ref_value=val, ref_lo=lo, ref_up=up
        )
        outside_range = df.filter(
            year=year, region=curr_region
        ).validate(
            criteria={ curr_variable: {
                'lo': lo,
                'up': up
            }},
            exclude_on_fail=True
        )

        util_filter_set_failed(df, curr_meta_column, outside_range, label='outside_range')

    # df.set_meta(flag_fail, name=meta_name_agg, index=df.filter(exclude=True))
    # df.set_meta(flag_pass, name=meta_name_agg, index=df.filter(exclude=False))
    df.set_meta(flag_fail, name=meta_name_agg)

    df.meta.loc[
        df.meta[curr_meta_columns].isin([flag_fail, flag_fail_missing]).all(axis=1),
        meta_name_agg] = flag_fail_missing

    df.meta.loc[
        df.meta[curr_meta_columns].isin([flag_pass, flag_pass_missing]).all(axis=1),
        meta_name_agg] = flag_pass_missing

    
    df.meta.loc[
        df.meta[curr_meta_columns].isin([flag_pass,]).all(axis=1),
        meta_name_agg] = flag_pass


    df.reset_exclude()


def filter_check_aggregate(df, variable, threshold, meta_name, meta_docs, key_historical=True, key_future=False, historical_columns=[], future_columns=[], ver='normal'):
    meta_name_agg = f'{meta_name}_aggregate'
    trs = 'approximately the' if ver=='teams' else ' within {:.0%} of'.format(threshold)
    meta_doc = f'Checks that the components of {variable} sum up to {trs} aggregate'
    util_filter_init(df, meta_name_agg, meta_doc, meta_docs, key_historical, key_future, historical_columns, future_columns)

    failed = df.check_aggregate(variable=variable, rtol=threshold)
    if failed is not None:
        max_rel_difference = ((failed['variable'] - failed['components']) / failed['variable']).max(level=[0,1,2,3])
        util_filter_save_value_column(df, meta_name_agg, max_rel_difference, value_columns=value_columns, plot=False)

    util_filter_set_failed(df, meta_name_agg, failed)
    df.reset_exclude()

def filter_validate(df, variable, missing_flag, year, lo, up, meta_name, key_historical=True, key_future=False,
                    historical_columns=[], future_columns=[], bound_threshold=1, ver='normal'):

    if bound_threshold != 1:
        lo = lo * (1-bound_threshold)
        up = up * (1+bound_threshold)


    if type(year) == str:
        # Interpret string as range:
        year = range(*[int(x) for x in year.split('-')])

    meta_name = f'{meta_name}' # delete validate
    trs = 'in {}'.format(year) if ver=='teams' else 'are within {} and {} in {}'.format(
        variable, lo, up, 'any year' if year is None else year
    )
    meta_doc = f'Checks that the values of {trs} '
    util_filter_init(df, meta_name, meta_doc, meta_docs, key_historical, key_future, historical_columns, future_columns)
    util_filter_check_exists(df, meta_name, variable, label=missing_flag)
    

    if type(year) == range:
        if lo is not None:
            util_filter_save_value_column(df, f'{variable} min', variable, value_columns=value_columns, year=year, method=np.min, ref_lo=lo)
        if up is not None:
            util_filter_save_value_column(df, f'{variable} max', variable, value_columns=value_columns, year=year, method=np.max, ref_up=up)
    else:
        util_filter_save_value_column(df, variable, variable, value_columns=value_columns, year=year, ref_lo=lo, ref_up=up)
    failed = df.filter(year=year).validate({variable: {'lo': lo, 'up': up}})

    util_filter_set_failed(df, meta_name, failed)

    df.reset_exclude()
    
    
def to_series(pyam_df):
    cols = pyam.YEAR_IDX
    return (
        pyam_df
        .data[cols+['value']]
        .set_index(cols)
        .value
    )    


# =============================================================================
# % increases
# =============================================================================

def change_year(series, year):
    # When calculating changes over time, the year index should match
    # to be able to subtract two series of different years
    new_series = series.reset_index()
    new_series['year'] = year
    return new_series.set_index(pyam.YEAR_IDX)['value']

def calc_increase_percentage(df, variable, year1, year2, suffix='|{}-{} change'):
    variable_year1 = to_series(df.filter(variable=variable, year=year1))
    variable_year2 = to_series(df.filter(variable=variable, year=year2))
    change_year1_year2 = 100*(
        change_year(variable_year2, year1) - variable_year1
    ) / variable_year1
    df.append(
        change_year1_year2,
        variable=variable+suffix.format(year1, year2), unit='% change',
        inplace=True
    )
    
    
def strip_version(model):
    model = model.replace('/','-')
    split = model.split(' ')
    
    if len(split) > 1:
        return ' '.join(split[:-1])
    return model    



# =============================================================================
# Merge sheet - pivot tables
# =============================================================================

def simple_pivot_cat_count(df, columns, index ):
    
    dfsp = df.pivot_table(
        columns=columns,
        index=index, 
        aggfunc=len,
        fill_value=0)
    dfsp['Sum'] = dfsp.sum(axis=1)

    return dfsp

def write_simple_pivot(writer, dfsp, sheet_name, header_format=None):
        
        dfsp.to_excel(
                writer,
                sheet_name=sheet_name,
                index=True, 
                header=True)
    
        workbook  = writer.book
        worksheet = writer.sheets[sheet_name]

        if header_format is None:
            header_format = workbook.add_format(
            {'bold': True, 
             'text_wrap': True, 
             'valign': 'top',
             'align': 'center'})
        for col_num, value in enumerate(dfsp.columns.values):
            worksheet.write(0, col_num+1, value, header_format)
            
        worksheet.set_column(0, len(dfsp.columns), 15, None)


# =============================================================================
#%% Summary iconics table
# Auxiliary functions for pyam
# =============================================================================

def year_of_net_zero(data, years, threshold):
    prev_val = 1 #0
    prev_yr = 1 #np.nan

    for yr, val in zip(years, data):
        if np.isnan(val):
            continue
        
        if val < threshold:
            x = (val - prev_val) / (yr - prev_yr) # absolute change per year
            return prev_yr + int((threshold - prev_val) / x) + 1 # add one because int() rounds down
        
        prev_val = val
        prev_yr = yr
    return np.inf


def get_from_meta_column(df, x, col):
    val = df.meta.loc[x.name[0:2], col]
    return val if val < np.inf else max(x.index)


def filter_and_convert(df, variable, unitin='', unitout='', factor=1):
    return (df
            .filter(variable=variable)
            .convert_unit(unitin, unitout, factor)
            .timeseries()
           )


def aggregate_missing_only(df, variable, components, append=False):
    missing = df.require_variable(variable=variable)
    if missing is not None:
        # missing = missing.loc[missing.model!='Reference',:]
        # if missing, aggregate energy and ip
        mapping = {}
        for model in missing.model.unique():
            mapping[model] = list(missing.loc[missing.model==model, 'scenario'])
        
        # Aggregate and add to the df
        if len(mapping)>0:
            for model, scenarios in mapping.items():
                try:
                    newdf =  df.filter(model=model, scenario=scenarios, variable=components).aggregate(variable=variable, components=components, append=False)
            
                    df.append(
                        newdf,
                    inplace=append
                    )
                except(IndexError):
                    print('No components:{},{}'.format(model, scenarios))
    return df

#%% Boxplot function

def plot_box_meta(dfb, varis, yticks=None, xlabel='', fname=None, palette=None):
    dfb1 = dfb.copy(deep=True)[varis]
    dfb1 = dfb1.reset_index().melt(id_vars=['model','scenario'], value_vars=dfb1.columns)
    sns.set_theme(style="ticks")
    figheight = len(varis)*0.5
    fig, ax = plt.subplots(figsize=(10, figheight))
    sns.boxplot(x='value', y='variable', data=dfb1, ax=ax,
                whis=[0,100], width=0.6, palette=palette)
    sns.stripplot(x='value', y='variable', data=dfb1, ax=ax,
                  size=4, color=".3", linewidth=0)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')

    # Fix yticklabels
    if type(yticks)==list:
        ax.set_yticklabels(yticks)
    
    ax.grid(True)
    
    plt.tight_layout()
    if type(fname)==str:
        fig.savefig(fname, dpi=300)#, bbox_inches=True)
        
        
# =============================================================================
#%% Write and format meta sheet for iconics table
# =============================================================================
def write_meta_sheet(df, filename, startfile=False):

    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    # Write data sheet
    df.to_excel(writer, sheet_name='data', include_meta=True)
    
    
    
    # Write meta page
    worksheet = writer.sheets['meta']
    worksheet.set_column(0, 0, 20, None)
    worksheet.set_column(1, 1, 25, None)
    worksheet.freeze_panes(1, 2)
    worksheet.autofilter(0, 0, len(df.meta), len(df.meta.columns)+1)
    
    # Write meta sheet headers
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
    
    # Write meta sheet conditional formatting
    
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
    
    # Write meta sheet value formatting
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
    if startfile:
        os.startfile(filename)