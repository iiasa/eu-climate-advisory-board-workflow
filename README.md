# EU Climate Advisory Board Scenario Explorer Workflow

Copyright 2022 IIASA

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository has the definitions for the **EU Climate Advisory Board Scenario Explorer**
used for submission of emissions scenarios to support the European Scientific Advisory Board on Climate Change
in its advice relating to the EU Climate Law.

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
