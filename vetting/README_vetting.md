# Information on vetting and analysis

For more documentation and explanatory notes, see section 3 of the 
[report](https://pure.iiasa.ac.at/18830).

Proceedure largely involves the following steps:
1. Prepare and process the input reference data. *(optional)*
2. Run the global and regional vetting scripts.
3. Merge the two outputs using `3_merge_summaries.py`.
4. Run `4_iconics_table.py` to produce data and metadata indicators.  


## Pre-processing of Reference data (optional - data included in repo)
Data has been bespokely prepared for both the global and regional vetting 
proceedures. This is stored in the file 'input_reference_all.csv'
To process the input reference data, use `regional/preprocess_input_ref_data.py`.
This is setup inteded for internal purposes - it has not been adapted for 
external (non-IIASA) use.  


## Vetting  
Vetting checks the scenarios against the input reference data in 
`input_data/input_reference_all.csv'.  

The vetting scripts should be run in this order:  
- Global: `1_scenario_vetting_global.py`  
- Regional: `2_scenario_vetting_regional.py`.  

The settings for the vetting are configured in:  
- `config_vetting_global.yaml`  
- `config_vetting_regional.yaml`  

Merge the two outputs using `3_merge_summaries.py`. This script takes summary 
columns from the two outputs above and can be used for defining what combinations 
of Global and Regional `Pass/Fail` will constitute an overall `PASS`.


## Vetting and auxiliary functions
The file `vetting_functions.py` contains functions used for the vetting,  
auxiliary functions to pyam for the calculation of metadata indicators 
(e.g. year of net zero, cumulative emissions), and outputs such as the formatted meta table and boxplots.


## Iconics data table and analysis  
A comprehensive metadata table was produced to assist the analaysis of the EUABCC.
This metadata has been added to the database. Two scripts are provided here for 
information.
- `4_extra_vars_meta_indicators.py`: Reads data from the database, and calculates
a number of additional variables and metadata indicators, which was subsequently
added to the databse.
- `5_iconics_table_db.py`: reads the data from the database and produces the 
dataset and metadata table in Excel, with conditional formatting, as provided to
EUABCC.
-  

The script `4_iconics_table.py` takes the input harmonized data, and calculates 
a number of calculations thatb are stored as `meta` that were used to assist 
analysis of the scenarios.



