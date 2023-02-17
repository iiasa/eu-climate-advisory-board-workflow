# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:01:43 2023

@author: byers
"""

# vetting_functions.py
import numpy as np
import pandas as pd
import pyam

#%% Define functions used to perform checks
log = True
print_log = print if log else lambda x: None
meta_docs = {}
historical_columns = []
future_columns = []
value_columns = {}


print_log = print if log else lambda x: None
###################################
#
# Define functions used to perform
# checks
#
###################################

def create_reference_df(df, ref_model, ref_scenario, check_variables, ref_year):
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

def util_filter_check_exists(df, meta_name, variable, year=None):
    var_doesnt_exist = pyam.require_variable(df, variable=variable, year=year, exclude_on_fail=True)
    if var_doesnt_exist is not None:
        missing_indices = var_doesnt_exist[['model', 'scenario']]
        df.set_meta('missing', meta_name, missing_indices)

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

def util_filter_save_value_column(df, name, variable_or_values, year=None, method=None, ref_value=None, ref_lo=None, ref_up=None, plot=True, **kwargs):
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

def filter_with_reference(df, ref_df, thresholds, year, meta_name, meta_docs, key_historical=True, key_future=False, historical_columns=[], future_columns=[], flag_fail='Fail', flag_pass='Pass', ver='normal'):
    meta_name_agg = f'{meta_name} summary'
    trs = '' if ver=='teams' else f': {thresholds}'
    util_filter_init(df, meta_name_agg, f'Checks that scenario is within reasonable range'+trs,
                     meta_docs, key_historical, key_future, historical_columns, future_columns)

    curr_meta_columns = []
    # Loop over each variable and value
    for idx, val in ref_df.iteritems():
        _, _, curr_region, curr_variable, curr_unit, *_ = idx

        valstr = np.round(val, 2)
        curr_meta_column = f'({meta_name} {year}: {curr_variable} = {valstr})'
        curr_meta_columns.append(curr_meta_column)
        df.set_meta(flag_pass, curr_meta_column)

        # Step 1: Identify scenarios where this variable/year is missing
        util_filter_check_exists(df, curr_meta_column, curr_variable, year)

        # Step 2: Identify scenarios where the value is out of range
        lo = val * (1-thresholds[curr_variable])
        up = val * (1+thresholds[curr_variable])
        util_filter_save_value_column(df, 
            curr_meta_column, curr_variable, year=year,
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
        df.meta[curr_meta_columns].isin([flag_pass, 'missing']).all(axis=1),
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
        util_filter_save_value_column(df, meta_name_agg, max_rel_difference, plot=False)

    util_filter_set_failed(df, meta_name_agg, failed)
    df.reset_exclude()

def filter_validate(df, variable, year, lo, up, meta_name, key_historical=True, key_future=False,
                    historical_columns=[], future_columns=[], bound_threshold=1, ver='normal'):

    if bound_threshold != 1:
        lo = lo * (1-bound_threshold)
        up = up * (1+bound_threshold)


    if type(year) == str:
        # Interpret string as range:
        year = range(*[int(x) for x in year.split('-')])

    meta_name = f'{meta_name} validate'
    trs = 'in {}'.format(year) if ver=='teams' else 'are within {} and {} in {}'.format(
        variable, lo, up, 'any year' if year is None else year
    )
    meta_doc = f'Checks that the values of {trs} '
    util_filter_init(df, meta_name, meta_doc, meta_docs, key_historical, key_future, historical_columns, future_columns)
    util_filter_check_exists(df, meta_name, variable)

    if type(year) == range:
        if lo is not None:
            util_filter_save_value_column(df, f'{variable} min', variable, year=year, method=np.min, ref_lo=lo)
        if up is not None:
            util_filter_save_value_column(df, f'{variable} max', variable, year=year, method=np.max, ref_up=up)
    else:
        util_filter_save_value_column(df, variable, variable, year=year, ref_lo=lo, ref_up=up)
    failed = df.filter(year=year).validate({variable: {'lo': lo, 'up': up}})

    util_filter_set_failed(df, meta_name, failed)

    df.reset_exclude()
    
    
def to_series(pyam_df):
    cols = pyam.utils.YEAR_IDX
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
    return new_series.set_index(pyam.utils.YEAR_IDX)['value']

def calc_increase_percentage(df, variable, year1, year2, suffix='|{}-{} change'):
    variable_year1 = to_series(df.filter(variable=variable, year=year1))
    variable_year2 = to_series(df.filter(variable=variable, year=year2))
    change_year1_year2 = (
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