# Information on vetting and analysis

For more documentation and explanatory notes, see section 3 of the 
[report](https://pure.iiasa.ac.at/18830).

Proceedure largely involves the following steps:
1. Process the input reference data. *(optional)*
2. Run the global and regional vetting scripts.
3. Merge the two outputs using `3_merge_summaries.py`.
4. Run `4_iconics_table.py` to produce data and metadata indicators.


## Pre-processing of Reference data  
Process the input reference data, using `regional/preprocess_input_ref_data.py`.


## Vetting  
Vetting checks the scenarios against the input reference data in 
`input_data/input_reference_all.csv'.  

The vetting scripts should be run in this order:  
Global: `global/1_scenario_vetting.py`  
Regional: `regional/2_scenario_vetting_regional.py`.  

The settings for the vetting are configured in:  
`config_vetting_global.yaml`  
`config_vetting_regional.yaml`  

Merge the two outputs using `3_merge_summaries.py`. This script takes summary 
columns from the two outputs above and can be used for defining what combinations 
of Global and Regional `Pass/Fail` will constitute an overall `PASS`.

## Iconics data table and analysis  
The script `4_iconics_table.py` takes the input harmonized data, and calculates 
a number of calculations thatb are stored as `meta` that were used to assist 
analysis of the scenarios.



