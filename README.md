# EU Climate Advisory Board Scenario Explorer Workflow

Copyright 2022 IIASA

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository has the definitions for the **EU Climate Advisory Board Scenario Explorer**
used for submission of emissions scenarios to support the European Scientific Advisory Board on Climate Change
in its advice relating to the EU Climate Law. It also contains scripts relating to the vetting, 
processing and analysis of scenarios to assist the identification of scenarios compliant 
with EU Climate Law and identification of Iconic Pathways.

Visit https://data.ece.iiasa.ac.at/eu-climate-advisory-board-submission for more information.


## Data submission

The scenario data has to be submitted as an xlsx file following the **IAMC data format**
via the **EU Climate Advisory Board Scenario Explorer** hosted by IIASA at
https://data.ece.iiasa.ac.at/eu-climate-advisory-board-submission.

### Model registration

Please read the instructions on model registration at the Scenario Explorer About-page
([link](https://data.ece.iiasa.ac.at/eu-climate-advisory-board-submission/#/about)).

### Variable and region definitions

The columns **region** and **variable** have to follow the codelists given in the folder 
[definitions](definitions).

## Analysis  
### Vetting and analysis (vetting folder)  
Vetting of baseline and near-term plausibility checks requires the preparation 
of Reference datasets against which to check the scenario submissions. Checks were 
done against a selection of Emissions and Energy variables for 2019. More information
including the order in which to run scripts is provided in the `README_vetting.md` file 
within that folder. 

For further documentation, see:  

Byers, E., Brutschin, E., Sferra, F., Luderer, G., Huppmann, D., Kikstra, J., 
Pietzcker, R., Rodrigues, R., & Riahi, K.  
Scenarios processing, vetting and feasibility assessment for the European Scientific 
Advisory Board on Climate Change.  
*International Institute for Applied Systems Analysis, Laxenburg,* 2023.  
[https://pure.iiasa.ac.at/18828](https://pure.iiasa.ac.at/18828)  

### Equity assessment
The analysis of emissions pathways taking into account equity considerations is 
published here.  
**Report:**  
Pelz, S., Rogelj, J., Riahi, K., 2023. Evaluating equity in European climate 
change mitigation pathways for the EU Scientific Advisory Board on Climate Change.  
*International Institute for Applied Systems Analysis, Laxenburg,* 2023.  
[https://pure.iiasa.ac.at/18830](https://pure.iiasa.ac.at/18830)

**Data and code:**  
[https://doi.org/10.5281/zenodo.7949883](https://doi.org/10.5281/zenodo.7949883)

## Workflow

The module `workflow.py` in this repository has a function `main(df: pyam.IamDataFrame) -> pyam.IamDataFrame:`.
It is used to validate any data submission to the Scenario Explorer against the project-specific codelists
and perform region-aggregation (optional).

## Dependencies

This repository uses the Python package **nomenclature** for scenario validation and region processing.
The nomenclature package provides a structured way to define code-lists for validation and mappings
for automated region-processing to support model comparison projects.
[Read the nomenclature docs](https://nomenclature-iamc.readthedocs.io) for more information...

<img src="https://github.com/IAMconsortium/pyam/raw/main/doc/logos/pyam-logo.png" width="133" height="100" align="right" alt="pyam logo" />

The nomenclature package depends on the Python package **pyam**.
The pyam package was developed to facilitate working with timeseries
data conforming to the IAMC structure. Features include scenario processing, plotting,
algebraic computations to derive indicators, validation of values, aggregation and downscaling of data,
and import/export with various file formats (`xlsx`, `csv`, frictionless-datapackage).
[Read the pyam docs](https://pyam-iamc.readthedocs.io) for more information...
