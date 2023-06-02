# ******************************************************************************
# * Base run script that was used to produce the figures for Byers et al. (2023) 
# * "Scenarios processing, vetting and feasibility assessment for the EU Scientific Advisory Board on Climate Change"
# *
# * Author: Jarmo S. Kikstra
# * Date last edited: June 2, 2023
# ******************************************************************************
# set working directory to root of this project
install.packages("here")
library("here")
# if(!require(rstudioapi)) { install.packages("rstudioapi"); require(rstudioapi)}
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # set work directory to this file
here::i_am("ClimateAssessmentRun2023 - EU CAB - visualisation only.Rproj") # look upwards from here to find the project file

source(here("scripts", "utils.R"))
load_pkgs(first.time = T)
set_data_paths()

# uncomment the line below for the first run (takes a bit less than 1 minute on Jarmo's laptop)
# prep_climatedata_eucab()

# Produce all figures

source(here("scripts", "table_stats.R")) # peak temperature stats
source(here("scripts", "figure_timeseries_temperature.R")) # temperature development, 3 panels (scenario ranges, climate ranges, combined ranges)

