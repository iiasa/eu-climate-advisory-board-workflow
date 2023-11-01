# ******************************************************************************
# * Script for table for temperature outcomes for Byers et al. (2023) 
# * "Scenarios processing, vetting and feasibility assessment for the EU Scientific Advisory Board on Climate Change"
# *
# * Author: Jarmo S. Kikstra
# * Date last edited: May 30, 2023
# ******************************************************************************

decimals = 2

# peak temperatures
# climate variables (peak):
# - 'p5 peak warming (MAGICCv7.5.3)'
# - 'median peak warming (MAGICCv7.5.3)'
# - 'p95 peak warming (MAGICCv7.5.3)'
# climate variables (2100):
# - `median warming in 2100 (MAGICCv7.5.3)`

table.peak.rawdata <- load_meta_eucab_clim_and_vet()

table.peak.data.scenariorange <- table.peak.rawdata %>% 
  group_by(Category, Category_name) %>% 
  summarise(
    `Scenario range p50` = round_decimal(median(`median peak warming (MAGICCv7.5.3)`), decimals),
    `Scenario range p5` = round_decimal(quantile(`median peak warming (MAGICCv7.5.3)`, probs=0.05), decimals),
    `Scenario range p95` = round_decimal(quantile(`median peak warming (MAGICCv7.5.3)`, probs=0.95), decimals)
  ) %>% 
  mutate(`Scenario range (peak)` = paste0(
    `Scenario range p50`, " [", `Scenario range p5`, "-", `Scenario range p95`, "]"
  )) %>% 
  select(Category, Category_name, `Scenario range (peak)`)
table.peak.data.scenariorange

table.2100.data.scenariorange <- table.peak.rawdata %>% 
  group_by(Category, Category_name) %>% 
  summarise(
    `Scenario range p50` = round_decimal(median(`median warming in 2100 (MAGICCv7.5.3)`), decimals),
    `Scenario range p5` = round_decimal(quantile(`median warming in 2100 (MAGICCv7.5.3)`, probs=0.05), decimals),
    `Scenario range p95` = round_decimal(quantile(`median warming in 2100 (MAGICCv7.5.3)`, probs=0.95), decimals)
  ) %>% 
  mutate(`Scenario range (2100)` = paste0(
    `Scenario range p50`, " [", `Scenario range p5`, "-", `Scenario range p95`, "]"
  )) %>% 
  select(Category, Category_name, `Scenario range (2100)`)
table.2100.data.scenariorange

table.peak.data.climaterange <- table.peak.rawdata %>% 
  group_by(Category, Category_name) %>% 
  summarise(
    `Climate range p50` = round_decimal(median(`median peak warming (MAGICCv7.5.3)`), decimals),
    `Climate range p5` = round_decimal(median(`p5 peak warming (MAGICCv7.5.3)`), decimals),
    `Climate range p95` = round_decimal(median(`p95 peak warming (MAGICCv7.5.3)`), decimals)
  ) %>% 
  mutate(`Climate range` = paste0(
    `Climate range p50`, " [", `Climate range p5`, "-", `Climate range p95`, "]"
  )) %>% 
  select(Category, Category_name, `Climate range`)
table.peak.data.climaterange

table.peak.data.combinedrange <- table.peak.rawdata %>% 
  group_by(Category, Category_name) %>% 
  summarise(
    
    # ED: other way: all scenarios, with all 3 temperatures, and then take the 5th or 95th percentile.
    
    `Combined range p50` = round_decimal(median(`median peak warming (MAGICCv7.5.3)`), decimals),
    `Combined range p5` = round_decimal(quantile(`p5 peak warming (MAGICCv7.5.3)`, probs=0.05), decimals),
    `Combined range p95` = round_decimal(quantile(`p95 peak warming (MAGICCv7.5.3)`, probs=0.95), decimals)
  ) %>% 
  mutate(`Combined range` = paste0(
    `Combined range p50`, " [", `Combined range p5`, "-", `Combined range p95`, "]"
  )) %>% 
  select(Category, Category_name, `Combined range`)
table.peak.data.combinedrange

# combine table

table.peak.data.all <- table.peak.data.scenariorange %>% 
  left_join(table.2100.data.scenariorange) %>% 
  left_join(table.peak.data.climaterange) %>% 
  left_join(table.peak.data.combinedrange)
table.peak.data.all

# add scenario number
scens.count <- table.peak.rawdata %>% 
  group_by(Category) %>% count() %>%
  mutate(`Category [# scenarios]` = paste0(
    Category, " [", n, "]"
  )) %>% 
  select(-n)

table.peak <- scens.count %>% 
  left_join(table.peak.data.all) %>% 
  ungroup() %>% 
  select(-Category)

# save

write_excel_csv(
  x = table.peak,
  file = here("analyses", "table_stats.csv"),
  delim = ",",
)
