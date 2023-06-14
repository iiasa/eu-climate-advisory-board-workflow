import logging
import os
import re
import sys
import tempfile
from pathlib import Path

import pyam
from nomenclature import DataStructureDefinition, RegionProcessor, process

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
log = logging.getLogger()
log.handlers.clear()
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)

here = Path(__file__).absolute().parent


def main(df: pyam.IamDataFrame) -> pyam.IamDataFrame:
    """Project/instance-specific workflow for scenario processing"""

    # Cast to an IamDataFrame if necessary
    if not isinstance(df, pyam.IamDataFrame):
        df = pyam.IamDataFrame(df)

    # Run the validation and region-processing
    definition = DataStructureDefinition(here / "definitions")
    processor = RegionProcessor.from_directory(path=here / "mappings", dsd=definition)
    df = process(df, definition, processor=processor)

    # Run MAGICC processing
    magicc_results = climate_assessment(df)

    # if magicc runs without any problems, add to the results
    if magicc_results:
        return df.append(magicc_results)
    else:
        return df


def climate_assessment(df):
    """Compute climate impact assessment"""
    # postprocessing with the MAGICC climate model
    # df.run_climate_model('MAGICC7')  # this is not yet implemented in pyam
    try:
        log.info("Importing import_timeseries.climate.run_magicc")
        import climate.run_magicc as magicc

        # create magicc input file - global emissions
        reg = re.compile(r"^Emissions[\|][^\|]*")
        emiss_vars = [x for x in df["variable"].unique() if reg.match(x)]
        emiss_df = df.filter(region="World", variable=emiss_vars)
        for scenario in emiss_df.scenario:
            magicc_df = emiss_df.filter(scenario=scenario)
            tmpdir = tempfile.mkdtemp(prefix="magicc_", dir=os.getcwd())
            emiss_file = os.path.join(tmpdir, "magicc_input.xlsx")
            log.info(f"magicc_input: {emiss_file}")
            # 'H:\\git\\ixmp_server_integration\\237_check\\magicc_input.xlsx'
            # magicc_log_file = '{}.log'.format(emiss_file)
            magicc_result_file = f"{emiss_file}.magicc.xlsx"
            magicc_log_file = f"{emiss_file}.log"
            magicc_df.to_excel(emiss_file)
            # run magicc
            log.info(f"call run_magicc({emiss_file})")
            magicc.run_magicc(emiss_file)
            # read and merge (append) magicc results
            if os.path.isfile(magicc_result_file):
                return pyam.IamDataFrame(magicc_result_file)
            elif os.path.isfile(magicc_log_file):
                # report magicc disgnostics errors
                for line in open(magicc_log_file, "r"):
                    if "magicc" in line:
                        log.error(line)
                return None
            else:
                log.error("run_magicc failed (neither result nor log file)")
                return None
    except ImportError:
        log.warning("no import_timeseries.climate.run_magicc file found!")
        return None
