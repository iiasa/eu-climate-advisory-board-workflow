# EU Climate Advisory Board Scenario Explorer Workflow

Copyright 2022 IIASA

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository has the definitions for the **EU Climate Advisory Board Scenario Explorer**
used for submission of emissions scenarios to support the European Scientific Advisory Board on Climate Change
in its advice relating to the EU Climate Law.

Visit https://data.ece.iiasa.ac.at/eu-climate-advisory-board-submission for more information.

### Model registration

If you want to register a model, please read the
[instruction](https://nomenclature-iamc.readthedocs.io/en/stable/user_guide/model-registration.html)
on the nomenclature documentation.

### Workflow

The module `workflow.py` has a function `main(df: pyam.IamDataFrame) -> pyam.IamDataFrame:`.

Per default, this function takes an **IamDataFrame** and returns it without
modifications. [Read the docs](https://pyam-iamc.readthedocs.io) for more information
about the **pyam** package for scenario analysis and data visualization.

**Important**: Do not change the name of the module `workflow.py` or the function `main`
as they are called like this by the Job Execution Service. Details can be found
[here](https://wiki.ece.iiasa.ac.at/wiki/index.php/Scenario_Explorer/Setup#Job_Execution_Service).

### Project nomenclature

The folder `definitions` can contain the project nomenclature, i.e., list of allowed
variables and regions, for use in the validation workflow. See the **nomenclature**
package for more information ([link](https://github.com/iamconsortium/nomenclature)).

The folder `mappings` can contain model mappings that are used to register models and
define how results should be processed upon upload to a Scenario Explorer.
