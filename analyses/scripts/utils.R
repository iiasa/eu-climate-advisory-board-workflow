# ******************************************************************************
# * This file hosts useful functions for producing the analysis
# * of the climate section of Byers et al. (2023) 
# * "Scenarios processing, vetting and feasibility assessment for the EU Scientific Advisory Board on Climate Change"
# *
# * Author: Jarmo S. Kikstra
# * Date last edited: May 30, 2023
# ******************************************************************************


# R packages and path utils ====

load_pkgs <- function(first.time = F) {
  # first.time. If FALSE, it will only load the necessary packages. If TRUE, it will also first install them.

  # if(!require(rstudioapi)) { install.packages("rstudioapi"); require(rstudioapi)}

  # could be replaced by p_load?

  pkgs <<- c(
    "vroom", # did anyone say "superspeed"? (for reading in csv files)
    "tidyverse", # main data-wrangling and visualisation package
    "readxl", # for reading excel files
    "ggthemes", # for some themes, especially theme_hc()
    # "here", # for working with relative paths; should already be loaded by the first lines of `produce_all_figures.R`
    "patchwork" # for easy arrangement of figures
  )

  # if necessary, first install all packages
  if (first.time) {
    install.packages(pkgs)
  }
  # load packages
  load <- lapply(pkgs, library, character.only = TRUE)

  # make sure we use the right function where two or more packages clash
  select <- dplyr::select # explicitly say that we mean dplyr's select function whenever we use select (not the one from the MASS library...)
  filter <- dplyr::filter # explicitly say that we mean dplyr's filter function whenever we use filter (not the one from the stats library...)
  mutate <- dplyr::mutate # explicitly say that we mean dplyr's mutate function whenever we use mutate
  map <- purrr::map # explicitly say that we mean purrr's mutate function whenever we use map (not the one from the maps package)
}


set_data_paths <- function(custom.path = NULL) {
  if (!is.null(custom.path)) {
    # set working directory to the custom path
    here::set_here(custom.path)
  } else if (is.null(custom.path)) {
    # set working directory to root of this project
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # set work directory to this file
    here::i_am("ClimateAssessmentRun2023 - EU CAB - visualisation only.Rproj") # look upwards from here to find the project file
  }



  # set locations for downloaded scenario data

  # vetting file taken from: https://iiasahub.sharepoint.com/sites/eceprog/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=CAHv2l&OR=Teams%2DHL&CT=1685436258479&clickparams=eyJBcHBOYW1lIjoiVGVhbXMtRGVza3RvcCIsIkFwcFZlcnNpb24iOiIyNy8yMzA1MDEwMDQyMiIsIkhhc0ZlZGVyYXRlZFVzZXIiOmZhbHNlfQ%3D%3D&cid=45801c38%2Dfd48%2D421a%2Dbc4b%2D4dc0c14d2c52&RootFolder=%2Fsites%2Feceprog%2FShared%20Documents%2FProjects%2FEUAB%2Fvetting&FolderCTID=0x012000AA9481BF7BE9264E85B14105F7F082FF
  # (as instructed by Edward Byers on Microsoft Teams)
  path.eucab.meta <<- here("data", "vetting_flags_global_regional_combined_20230512_v3.xlsx") 

  # set locations for other data files

  path.ipcc.colour.palette <<- here("data", "ipcc_color_palette.xlsx")

  # also set the path for where to save out the figures

  path.figures <<- here("analyses")
}


# IAMC style utils ====
iamc_wide_to_long <- function(df, upper.to.lower = F,
                              cols.to.keep = NULL) {

  # function assumes all five basic IAMC columns are there, and nothing more

  if (upper.to.lower) {
    df <- df %>%
      rename(
        model = Model,
        scenario = Scenario,
        region = Region,
        variable = Variable,
        unit = Unit
      )
  }

  if ((!is.null(cols.to.keep))){
    
    if(identical(
      df %>% distinct() %>% nrow(),
      df %>% select(all_of(cols.to.keep)) %>% distinct() %>% nrow()
    )){
      df <- df %>% select(all_of(cols.to.keep))  
    } else {
      error(
       "This code is dropping meta columns that would lead to duplicates. Please revise your code or look at your data." 
      ) 
    }
  }
    
  first.year <- colnames(df)[6] # assumes all five basic IAMC columns are there, and nothing more (also no meta-data)
  last.year <- colnames(df)[length(colnames(df))]

  df <- df %>%
    pivot_longer(
      cols = first.year:last.year,
      names_to = "year",
      values_to = "value"
    ) %>%
    drop_na(value) %>%
    mutate(year = as.numeric(year))

  return(df)
}

filter_wildcard_var <- function(df, variable.string,
                                lowercase.var.variable = F) {
  # accepts 4 input options:
  # - variable string
  # - variable string with an wildcard (*) at the end
  # - variable string with an wildcard (*) at the start
  # - variable string with wildcards (*) both at the start and the end

  if (!grepl(x = variable.string, pattern = "\\*")) {
    print("Filtering on the exact string, no wildcard.")
    if (lowercase.var.variable) {
      df <- df %>% filter(variable == variable.string)
    } else {
      df <- df %>% filter(Variable == variable.string)
    }
  } else {
    if (lowercase.var.variable) {

      # test for some exceptions
      if (length(str_split(string = variable.string, pattern = "\\*", n = Inf, simplify = FALSE)[[1]]) > 3) {
        return(print("This input string is not supported for this function. Do you have too many wildcards?"))
      }
      if (!(startsWith(variable.string, "*") | endsWith(variable.string, "*"))) {
        return(print("This input string is not supported for this function. Did you put a wildcard in the string, but not place it at the beginning or the end?"))
      }

      # do the filtering
      if (startsWith(variable.string, "*") & endsWith(variable.string, "*")) {
        print("Filtering variable based on prefix and suffix wildcards")
        df <- df %>% filter(grepl(
          x = variable,
          pattern = str_split(string = variable.string, pattern = "\\*", n = Inf, simplify = FALSE)[[1]][2] %>% as.character(),
          fixed = T
        ))
      } else if (startsWith(variable.string, "*")) {
        print("Filtering variable based on prefix wildcard")
        df <- df %>% filter(grepl(
          x = variable,
          pattern = vs,
          fixed = T
        ) & (
          grepl(x = substr(variable, nchar(variable) - nchar(vs) + 1, nchar(variable)), pattern = vs, fixed = T)
        ))
      } else if (endsWith(variable.string, "*")) {
        print("Filtering variable based on suffix wildcard")
        vs <- str_split(string = variable.string, pattern = "\\*", n = Inf, simplify = FALSE)[[1]][1] %>% as.character()
        df <- df %>% filter(grepl(
          x = variable,
          pattern = vs,
          fixed = T
        ) & (
          grepl(x = substr(Variable, 1, nchar(vs)), pattern = vs, fixed = T)
        ))
      } else {
        return(print("This input string is not supported for this function. Did you maybe put a wildcard not at the end or the beginning of the variable.string?"))
      }
    } else {

      # test for some exceptions
      if (length(str_split(string = variable.string, pattern = "\\*", n = Inf, simplify = FALSE)[[1]]) > 3) {
        return(print("This input string is not supported for this function. Do you have too many wildcards?"))
      }

      # do the filtering
      if (startsWith(variable.string, "*") & endsWith(variable.string, "*")) {
        print("Filtering Variable based on prefix and suffix wildcards")
        df <- df %>% filter(grepl(
          x = Variable,
          pattern = str_split(string = variable.string, pattern = "\\*", n = Inf, simplify = FALSE)[[1]][2] %>% as.character(),
          fixed = T
        ))
      } else if (startsWith(variable.string, "*")) {
        print("Filtering Variable based on prefix wildcard")
        vs <- str_split(string = variable.string, pattern = "\\*", n = Inf, simplify = FALSE)[[1]][2] %>% as.character()
        df <- df %>% filter(grepl(
          x = Variable,
          pattern = vs,
          fixed = T
        ) & (
          grepl(x = substr(Variable, nchar(Variable) - nchar(vs) + 1, nchar(Variable)), pattern = vs, fixed = T)
        ))
      } else if (endsWith(variable.string, "*")) {
        print("Filtering Variable based on suffix wildcard")
        vs <- str_split(string = variable.string, pattern = "\\*", n = Inf, simplify = FALSE)[[1]][1] %>% as.character()
        df <- df %>% filter(grepl(
          x = Variable,
          pattern = vs,
          fixed = T
        ) & (
          grepl(x = substr(Variable, 1, nchar(vs)), pattern = vs, fixed = T)
        ))
      } else {
        return(print("This input string is not supported for this function. Did you maybe put a wildcard not at the end or the beginning of the variable.string?"))
      }
    }
  }



  return(df)
}

# AR6-like utils ====
rename_c1 <- function(df){
  df %>% mutate(Category=ifelse(Category%in%c("C1a","C1b"),"C1",Category)) %>%
    mutate(Category_name=ifelse(Category%in%c("C1"),"C1: Below 1.5°C with no or low OS",Category_name)) %>% 
    return(.)
}

rename_remind_and_ngfs <- function(df){
  remind_3_2.thatusedtobe.remind_3_1 <- tribble(
    ~old.model.name, ~new.model.name, ~scenario,
    "REMIND 3.1", "REMIND 3.2", "NZero_bioLim12_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "NZero_bioLim7p5_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "NZero_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "def_300_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "def_500_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "def_800_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "def_bioLim12_300_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "def_bioLim12_500_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "def_bioLim12_800_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "def_bioLim7p5_300_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "def_bioLim7p5_500_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "def_bioLim7p5_800_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "flex_300_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "flex_500_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "flex_800_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "flex_bioLim12_300_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "flex_bioLim12_500_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "flex_bioLim12_800_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "flex_bioLim7p5_300_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "flex_bioLim7p5_500_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "flex_bioLim7p5_800_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "rigid_300_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "rigid_500_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "rigid_800_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "rigid_bioLim12_300_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "rigid_bioLim12_500_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "rigid_bioLim12_800_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "rigid_bioLim7p5_300_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "rigid_bioLim7p5_500_withICEPhOP",
    "REMIND 3.1", "REMIND 3.2", "rigid_bioLim7p5_800_withICEPhOP",
  )
  
  ngfs.rename.unicode <- tribble(
    ~model, ~old.scenario.name, ~new.scenario.name,
    "GCAM 5.3+ NGFS", "NGFS-Below 2°C", "NGFS-Below 2C",
    "GCAM 5.3+ NGFS", "NGFS-Delayed transition", "NGFS-Delayed Transition",
    "MESSAGEix-GLOBIOM 1.1-M-R12", "NGFS-Below 2°C", "NGFS-Below 2C",
    "MESSAGEix-GLOBIOM 1.1-M-R12", "NGFS-Delayed transition", "NGFS-Delayed Transition",
    "REMIND-MAgPIE 3.0-4.4", "NGFS-Below 2°C", "NGFS-Below 2C",
    "REMIND-MAgPIE 3.0-4.4", "NGFS-Delayed transition", "NGFS-Delayed Transition"
  )
  
  df <- df %>% 
    left_join(
      ngfs.rename.unicode %>% rename(scenario=old.scenario.name)
    ) %>% 
    left_join(
      remind_3_2.thatusedtobe.remind_3_1 %>% rename(model=old.model.name)
    ) %>% 
    mutate(
      model=ifelse(!is.na(new.model.name),new.model.name,model),
      scenario=ifelse(!is.na(new.scenario.name),new.scenario.name,scenario)
    ) %>% 
    select(
      -new.model.name,
      -new.scenario.name
    )
  
  return(df)
}


prep_climatedata_eucab <- function(){
  
  meta.categories.and.versions <- load_meta_eucab_clim_and_vet() %>%
    select(model,scenario,Category,version)
  
  # load last run first, combine with meta, append to meta.categories.and.versions
  # then, same for penultimate run...
  # then, the one before...
  # etc...
  
  # load ClimateAssessmentRun2023 - EU CAB - additional REMIND version 3 and ECEMF (run_20230505)
  data.step1 <-  
      read_excel(
        path = here("..","ClimateAssessmentRun2023 - EU CAB - additional REMIND version 3 and ECEMF","run_20230505","output","remind_late_and_ecmf_emissions_alloutput.xlsx"),
        sheet = "data"
      ) %>% filter(grepl(Variable,pattern="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|",fixed=T)) %>% 
        select(-`1990`,-`2110`) %>% # empty lgl columns we don't need (temperature only reported for 1995-2100, so it's a remainder of emissions data)
        left_join(
          read_excel(
            path = here("..","ClimateAssessmentRun2023 - EU CAB - additional REMIND version 3 and ECEMF","run_20230505","output","remind_late_and_ecmf_emissions_alloutput.xlsx"),
            sheet = "meta"
          ) %>% rename_c1()
        ) %>% 
        rename(model=Model, scenario=Scenario)
  data.step1 <- data.step1 %>% rename_remind_and_ngfs()
  
  # load ClimateAssessmentRun2023 - EU CAB - additional REMIND version 2 (run_20230428)
  data.step2 <- 
      read_excel(
        path = here("..","ClimateAssessmentRun2023 - EU CAB - additional REMIND version 2","run_20230428","output","remind_sensitivity_emissions_alloutput.xlsx"),
        sheet = "data"
      ) %>% filter(grepl(Variable,pattern="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|",fixed=T)) %>% 
        left_join(
          read_excel(
            path = here("..","ClimateAssessmentRun2023 - EU CAB - additional REMIND version 2","run_20230428","output","remind_sensitivity_emissions_alloutput.xlsx"),
            sheet = "meta"
          ) %>% rename_c1()
        ) %>% 
        rename(model=Model, scenario=Scenario)
  data.step2 <- data.step2 %>% rename_remind_and_ngfs()

  # load ClimateAssessmentRun2023 - EU CAB - additional REMIND (run_20230415)
  data.step3 <- 
    read_excel(
      path = here("..","ClimateAssessmentRun2023 - EU CAB - additional REMIND","run_20230415","output","AdvisoryBoard_additional_remind_alloutput.xlsx"),
      sheet = "data"
    ) %>% filter(grepl(Variable,pattern="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|",fixed=T)) %>% 
    left_join(
      read_excel(
        path = here("..","ClimateAssessmentRun2023 - EU CAB - additional REMIND","run_20230415","output","AdvisoryBoard_additional_remind_alloutput.xlsx"),
        sheet = "meta"
      ) %>% rename_c1()
    ) %>% 
    rename(model=Model, scenario=Scenario)
  data.step3 <- data.step3 %>% rename_remind_and_ngfs()
  
  # load ClimateAssessmentRun2023 - EU CAB (run_20230213 [NOT run_20230210])
  data.step4 <- 
    vroom(
      here("..","ClimateAssessmentRun2023 - EU CAB","run_20230213","output","EU_CAB_climate_alloutputfiles", "EU_CAB_World_Emissions_alloutput.csv")
    ) %>% filter(grepl(Variable,pattern="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|",fixed=T)) %>% 
    rename(model=Model, scenario=Scenario) %>% 
    select(-`1990`,-`2110`,-`2130`,-`2150`) %>% # empty lgl columns we don't need (temperature only reported for 1995-2100, so it's a remainder of emissions data)
    left_join(
      read_excel(
        path = here("..","ClimateAssessmentRun2023 - EU CAB","run_20230213","output","EU_CAB_climate_alloutputfiles", "EU_CAB_World_Emissions_meta.xlsx"),
        sheet = "meta"
      ) %>% rename_c1()
    )
  data.step4 <- data.step4 %>% rename_remind_and_ngfs()
  
  
  # check if columns are equal in all data sets
  identical(data.step1 %>% colnames() %>% sort(),
            data.step2 %>% colnames() %>% sort())
  identical(data.step2 %>% colnames() %>% sort(),
            data.step3 %>% colnames() %>% sort())
  identical(data.step3 %>% colnames() %>% sort(),
            data.step4 %>% colnames() %>% sort())
  
  # data.step4 has more columns...
  ds.cols.st4 <- data.step4 %>% colnames() %>% sort()
  del.ds.cols.st4 <-  ds.cols.st4[!(ds.cols.st4 %in% (data.step3 %>% colnames() %>% sort()))]
  # check if there are no duplicates if deleting columns
  identical(data.step4 %>% nrow(), 
              data.step4 %>% select(-all_of(del.ds.cols.st4)) %>% distinct() %>% nrow())
  # delete data.step4 columns 
  data.step4 <- data.step4 %>% select(-all_of(del.ds.cols.st4))
  
  
  
  # combine
  df.step1 <- meta.categories.and.versions %>% left_join(data.step1) %>% drop_na(Variable)
  df.step2 <- meta.categories.and.versions %>% left_join(data.step2) %>% drop_na(Variable)
  df.step3 <- meta.categories.and.versions %>% left_join(data.step3) %>% drop_na(Variable)
  df.step4 <- meta.categories.and.versions %>% left_join(data.step4) %>% drop_na(Variable)
  
  df.all <- df.step1 %>% mutate(step=1) %>%  
    bind_rows(df.step2 %>% mutate(step=2)) %>% 
    bind_rows(df.step3 %>% mutate(step=3)) %>%
    bind_rows(df.step4 %>% mutate(step=4))
  
  # check whether there are duplicates, and if so, whether they come from different steps/files
  identical(
    df.all %>% select(model,scenario,Variable),
    df.all %>% select(model,scenario,Variable) %>% distinct()
  )
  identical(
    df.all %>% select(model,scenario,Variable,step),
    df.all %>% select(model,scenario,Variable,step) %>% distinct()
  )
   
  # keep newest duplicate
  duplicates <- df.all %>% select(model,scenario,step) %>% distinct() %>% 
    group_by(model,scenario) %>% count() %>% filter(n>1) # 4 scenarios with duplication
  duplicates
  # keep lowest step number 
  duplicates <- duplicates %>% left_join(
    df.all %>% left_join(duplicates) %>% filter(!is.na(n)) %>% distinct(model,scenario,step) %>% group_by(model,scenario) %>% 
      summarise(
        keep.step = min(step)
      )
  )
  
  df.all.noduplicates <- 
    bind_rows(
      df.all %>% left_join(duplicates) %>% filter(is.na(n)) %>% select(-n,-keep.step) # 359 scenarios without duplication
    ) %>% 
    bind_rows(
      # 4 scenarios with duplication
      df.all %>% left_join(duplicates) %>% filter(!is.na(n)) %>% 
        filter(step==keep.step) %>% select(-n,-keep.step)
    )
  # check if correct now
  identical(
    df.all.noduplicates %>% select(model,scenario,Variable) %>% arrange(model,scenario,Variable),
    df.all %>% select(model,scenario,Variable) %>% distinct() %>% arrange(model,scenario,Variable)
  )
  
  # check if the number of scenarios is the same as expected
  identical(meta.categories.and.versions %>% distinct(model,scenario) %>% count(),
            df.all.noduplicates %>% distinct(model,scenario) %>% count())
  # what scenarios are we still missing
  meta.categories.and.versions %>% distinct(model,scenario) %>% left_join(
    df.all.noduplicates %>% distinct(model,scenario) %>% mutate(data="available")
  ) %>% 
    filter(is.na(data)) %>% 
    write_excel_csv(
      x = .,
      here("data", "eucab_temperaturedata_missing.csv"),
      delim = ","
    )
  
  
  write_excel_csv(
    x = df.all.noduplicates,
    file = here("data", "eucab_temperaturedata_prepped.csv"), 
    delim = ","
  )
}

load_var_eucab <- function(variable,
                         with.lowercase.columns = F,
                         prep.climate.data.firsttime = F) {

  # This reads in the (very) big EVERYTHING file.
  # To keep the code for this repository simple, we stick to that, even though one could speed up this reproduction code by choosing to load in the standard AR6DB snapshot when the desired variable(s) is not unique to the EVERYTHING file [could check against some variable lists].

  
  
  if (prep.climate.data.firsttime) {
    prep_climatedata_eucab()
  }
  
  df <- vroom(
    here("data", "eucab_temperaturedata_prepped.csv")
  ) %>% rename(Model=model,Scenario=scenario)

  gc()


  if (grepl(pattern = "\\*", x = variable)) {
    df <- filter_wildcard_var(
      df,
      variable.string = variable
    )
  } else {
    df <- df %>%
      filter(
        Variable == variable
      )
  }
  # note: this function could be extended add the option to pass a list, which would be faster than the wildcard filter


  gc()
  return(df %>%
    iamc_wide_to_long(upper.to.lower = T,
                      cols.to.keep = c(
                        "model",
                        "scenario",
                        "region",
                        "variable",
                        "unit",
                        as.character(
                          seq(1995,2100)
                        )
                      )
                      ))
}

load_meta_eucab <- function(meta.sheet =  "Vetting_flags", # "meta",
                          make.lowercase.columns = F) {
  if (!make.lowercase.columns) {
    return(
      read_excel(
        path.eucab.meta,
        sheet = meta.sheet
      )
    )
  } else if (make.lowercase.columns) {
    return(
      read_excel(
        path.eucab.meta,
        sheet = meta.sheet
      ) %>%
        rename(model = Model, scenario = Scenario)
    )
  }
}

load_meta_eucab_clim_and_vet <- function(...){
  load_meta_eucab(meta.sheet = "Vetting_flags") %>% rename_c1() %>%
    filter(Category!="no-climate-assessment") %>% 
    filter(OVERALL_binary=="PASS") %>% 
    left_join(
      load_meta_eucab(meta.sheet = "meta_climate") %>% rename_c1() %>%
        filter(Category!="no-climate-assessment")
    ) %>% 
    return(.)
}


add_climate_emulator_col <- function(df) {

  # function could be made smarter with a more flexible filtering mechanism
  # currently assumes all columns to be entirely lower cap

  df <- df %>%
    mutate(
      emulator =
        ifelse(
          grepl(
            x = variable, pattern = "CICERO-SCM", fixed = T
          ),
          "CICERO",
          ifelse(
            grepl(
              x = variable, pattern = "MAGICCv7.5.3", fixed = T
            ),
            "MAGICC",
            ifelse(
              grepl(
                x = variable, pattern = "FaIRv1.6.2", fixed = T
              ),
              "FaIR",
              NA
            )
          )
        )
    ) %>%
    mutate(
      variable =
        ifelse(
          grepl(
            x = variable, pattern = "CICERO-SCM", fixed = T
          ),
          str_remove(string = variable, pattern = "CICERO-SCM\\|"),
          ifelse(
            grepl(
              x = variable, pattern = "MAGICCv7.5.3", fixed = T
            ),
            str_remove(string = variable, pattern = "MAGICCv7.5.3\\|"),
            ifelse(
              grepl(
                x = variable, pattern = "FaIRv1.6.2", fixed = T
              ),
              str_remove(string = variable, pattern = "FaIRv1.6.2\\|"),
              variable
            )
          )
        )
    )

  return(df)
}


# Calculations utils ====
round_decimal <- function(x, k) round(x = x, digits = k)


# Plotting utils ====
get_ipcc_colours <- function(type) {
  # note: this information is also in the eucab meta file, columns c(IMP_color_rgb,	IMP_color_hex, Category_color_rgb)

  if (type == "c.hex") {
    return(
      paste0(
        "#",
        read_excel(
          path = path.ipcc.colour.palette,
          sheet = "categories"
        ) %>%
          pull(Category_color_hex)
      )
    )
  }
  if (type == "c.list") {
    return(
      read_excel(
        path = path.ipcc.colour.palette,
        sheet = "categories"
      ) %>%
        pull(Category)
    )
  }
  if (type == "imp.hex") {
    return(
      paste0(
        "#",
        read_excel(
          path = path.ipcc.colour.palette,
          sheet = "imps"
        ) %>%
          pull(IMP_color_hex)
      )
    )
  }
  if (type == "imp.list") {
    return(
      read_excel(
        path = path.ipcc.colour.palette,
        sheet = "imps"
      ) %>%
        pull(imp)
    )
  }
  if (type == "ssp.hex") {
    return(
      paste0(
        "#",
        read_excel(
          path = path.ipcc.colour.palette,
          sheet = "ssps"
        ) %>%
          pull(ssp)
      )
    )
  }
  if (type == "ssp.list") {
    return(
      read_excel(
        path = path.ipcc.colour.palette,
        sheet = "ssps"
      ) %>%
        pull(ssp_color_hex)
    )
  }
}



# Code development utils ====
clean_code_style <- function(first.time = F) {
  if (first.time) {
    install.packages("styler")
  }
  library(styler)
  styler::style_dir()
}
