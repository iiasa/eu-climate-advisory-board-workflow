# ------------------------------------------------------------------------------
# R libraries
# ------------------------------------------------------------------------------

# JDBC connection
require(RJDBC)
# load xlsx package for data export to Excel
require(xlsx)
require(dplyr) #, warn.conflicts=F)
require(tidyr)


# ------------------------------------------------------------------------------
# environment variables
# ------------------------------------------------------------------------------

# IEA database connection
#drv<-JDBC("oracle.jdbc.driver.OracleDriver","/Users/jewell/Library/Application Support/ojdbc5.jar")
#drv <- JDBC("oracle.jdbc.driver.OracleDriver","C:/krey/eclipse/workspace/IamcDatabase_new/lib/ojdbc5.jar")
drv<-JDBC("oracle.jdbc.driver.OracleDriver","P:/ene.model/Rcodes/Common_files/ojdbc6.jar")


# working directory
osUser <- Sys.info()['login']
if (osUser == 'kolp') {
  wrkDir <- "D:/checks/IEA_2023"
} else if (osUser == 'krey') {
  wrkDir <- "C:/krey/TEMP"
} else {
  q()
}

pe_map <- data.frame(
  'FUEL' = c('Coal|Total','Oil|Total','Gas|Total','Biomass|Total',
           'Hydro|Total','Wind|Total','Ocean|Total','Other|Total' ,
           'Nuclear|Total','Geothermal|Total','Solar|Total'
           ,'Total', 'Fossil|Total') ,
  'VARIABLE' = c('Primary Energy|Coal','Primary Energy|Oil','Primary Energy|Gas','Primary Energy|Biomass',
             'Primary Energy|Hydro','Primary Energy|Wind','Primary Energy|Ocean','Primary Energy|Other',
             'Primary Energy|Nuclear','Primary Energy|Geothermal','Primary Energy|Solar',
             'Primary Energy','Primary Energy|Fossil'
             )
  )

ele_map <- data.frame(
  'FUEL' = c('Coal|Total','Oil|Total','Gas|Total','Biomass|Total',
             'Hydro|Total','Wind|Total','Ocean|Total','Other|Total' ,
             'Nuclear|Total','Geothermal|Total','Solar|Total'
             ,'Total', 'Fossil|Total') ,
  'VARIABLE' = c('Secondary Energy|Electricity|Coal','Secondary Energy|Electricity|Oil','Secondary Energy|Electricity|Gas','Secondary Energy|Electricity|Biomass',
                 'Secondary Energy|Electricity|Hydro','Secondary Energy|Electricity|Wind','Secondary Energy|Electricity|Ocean','Secondary Energy|Electricity|Other',
                 'Secondary Energy|Electricity|Nuclear','Secondary Energy|Electricity|Geothermal','Secondary Energy|Electricity|Solar',
                 'Secondary Energy|Electricity','Secondary Energy|Electricity|Fossil'
  )
)

fetot_map <- data.frame(
  'FUEL' = c('Coal|Total','Oil|Total','Gas|Total','Biomass|Total'
             ,'Solar|Total' ,'Heat|Total' ,'Geothermal|Total'
             ,'Total', 'Fossil|Total' ,'Electricity|Total') ,
  'VARIABLE' = c('Final Energy|Solids|Coal','Final Energy|Liquids','Final Energy|Gases','Final Energy|Solids|Biomass',
                 'Final Energy|Solar', 'Final Energy|Heat', 'Final Energy|Geothermal',
                 'Final Energy','Final Energy|Fossil' ,'Final Energy|Electricity'
  )
)

feind_map <- data.frame(
  'FUEL' = c('Coal|Total','Oil|Total','Gas|Total','Biomass|Total'
             ,'Solar|Total' ,'Heat|Total' ,'Geothermal|Total'
             ,'Total', 'Fossil|Total' ,'Electricity|Total') ,
  'VARIABLE' = c('Final Energy|Industry|Solids|Coal','Final Energy|Industry|Liquids','Final Energy|Industry|Gases','Final Energy|Industry|Solids|Biomass',
                 'Final Energy|Industry|Solar', 'Final Energy|Industry|Heat', 'Final Energy|Industry|Geothermal',
                 'Final Energy|Industry','Final Energy|Industry|Fossil' ,'Final Energy|Industry|Electricity'
  )
)

ferc_map <- data.frame(
  'FUEL' = c('Coal|Total','Oil|Total','Gas|Total','Biomass|Total'
             ,'Solar|Total' ,'Heat|Total' ,'Geothermal|Total'
             ,'Total', 'Fossil|Total' ,'Electricity|Total') ,
  'VARIABLE' = c('Final Energy|Residential and Commercial|Solids|Coal','Final Energy|Residential and Commercial|Liquids','Final Energy|Residential and Commercial|Gases','Final Energy|Residential and Commercial|Solids|Biomass',
                 'Final Energy|Residential and Commercial|Solar', 'Final Energy|Residential and Commercial|Heat', 'Final Energy|Residential and Commercial|Geothermal',
                 'Final Energy|Residential and Commercial','Final Energy|Residential and Commercial|Fossil' ,'Final Energy|Residential and Commercial|Electricity'
  )
)

fetrp_map <- data.frame(
  'FUEL' = c('Coal|Total','Oil|Total','Gas|Total','Biomass|Total'
             ,'Solar|Total' ,'Heat|Total' ,'Geothermal|Total'
             ,'Total', 'Fossil|Total' ,'Electricity|Total') ,
  'VARIABLE' = c('Final Energy|Transportation|Solids|Coal','Final Energy|Transportation|Liquids','Final Energy|Transportation|Gases','Final Energy|Transportation|Solids|Biomass',
                 'Final Energy|Transportation|Solar', 'Final Energy|Transportation|Heat', 'Final Energy|Transportation|Geothermal',
                 'Final Energy|Transportation','Final Energy|Transportation|Fossil' ,'Final Energy|Transportation|Electricity'
  )
)

fene_map <- data.frame(
  'FUEL' = c('Coal|Total','Oil|Total','Gas|Total'
             ,'Biomass|Total' ,'Total', 'Fossil|Total' ) ,
  'VARIABLE' = c('Final Energy|Non-Energy Use|Coal','Final Energy|Non-Energy Use|Oil'
                 ,'Final Energy|Non-Energy Use|Gas','Final Energy|Non-Energy Use|Biomass',
                 'Final Energy|Non-Energy Use','Final Energy|Non-Energy Use|Fossil' 
  )
)

setwd(wrkDir)

revision <- 2023
dataTable <- 'edb_data_2023'
regionScheme <- 'IPCCSR15'
fuelScheme <- 'OTHER'
lastOutputYear <- 2020
#adapted query for import volume
#!! Problem that this groups all data for the same region name but different models in the same category.
if (T) {
 peQuery <- paste("
  SELECT d.ryear, e.fuel, Sum(d.VALUE*u.FACTOR) EJ , u.iso
  FROM edb_unit_conversion u, ((edb_flow f 
  INNER JOIN ",dataTable," d ON f.CODE=d.FLOW_CODE)  
  INNER JOIN edb_fuel e ON e.PROD_CODE = d.PROD_CODE)   
  INNER JOIN edb_country c ON d.COUNTRY_CODE=c.CODE 
  inner join un_country_mapping u on u.country_name = c.name
  inner join un_iso_iana_itu_codes i on i.iso3 = u.iso
  WHERE ((d.ryear>=1971 And d.ryear<=",lastOutputYear,")  
  And d.rev_code=",revision," And d.UNIT=u.UNIT_IN 
  And u.UNIT_OUT='EJ')  
  And e.scheme='",fuelScheme,"'
  And ( ((e.fuel in ('Total')  )  And (f.name='total primary energy supply') ) )
  GROUP BY d.ryear, e.fuel ,u.iso  ORDER by d.ryear, e.fuel
 ", sep = '')
 
 eleGenQuery <- paste("
  SELECT d.ryear, e.fuel, Sum(d.VALUE*u.FACTOR) EJ , u.iso
   FROM edb_unit_conversion u, ((edb_flow f 
   INNER JOIN ",dataTable," d ON f.CODE=d.FLOW_CODE)  
   INNER JOIN edb_fuel e ON e.PROD_CODE = d.PROD_CODE)   
   INNER JOIN edb_country c ON d.COUNTRY_CODE=c.CODE 
  inner join un_country_mapping u on u.country_name = c.name
  inner join un_iso_iana_itu_codes i on i.iso3 = u.iso
   WHERE ((d.ryear>=1971 And d.ryear<=",lastOutputYear,")  
   And d.rev_code=",revision," And d.UNIT=u.UNIT_IN 
   And u.UNIT_OUT='EJ')  
   And e.scheme='",fuelScheme,"'
   And e.fuel in ('Wind|Total' ,'Nuclear|Total' ,'Solar|Total') 
   And (f.name='elect.output in gwh')
  GROUP BY d.ryear, e.fuel ,u.iso ORDER by d.ryear, e.fuel
  ", sep = '')
# And e.fuel in ('Coal|Total' ,'Oil|Total' ,'Gas|Total' ,'Biomass|Total','Hydro|Total' ,'Wind|Total' ,'Ocean|Total' ,'Other|Total' ,'Nuclear|Total' ,'Geothermal|Total' ,'Solar|Total','Total','Fossil|Total') 
 
  ieadb.conn <- dbConnect(drv, "jdbc:oracle:thin:@x8oda.iiasa.ac.at:1521/pGP3.iiasa.ac.at", "iea", "iea")
  data.pe <- dbGetQuery(ieadb.conn, peQuery)
  data.eleGen <- dbGetQuery(ieadb.conn, eleGenQuery)
  dbDisconnect(ieadb.conn)
  
 data.pe %>% 
    left_join(pe_map) %>% 
    mutate(UNIT='EJ/yr', MODEL='History', SCENARIO='IEA Energy Statistics (r2022)') %>%
    select(MODEL,SCENARIO,ISO,VARIABLE,UNIT,RYEAR,EJ) %>% 
    spread(RYEAR, EJ) -> histData.pe

 data.eleGen %>% 
   left_join(ele_map) %>% 
   mutate(UNIT='EJ/yr', MODEL='History', SCENARIO='IEA Energy Statistics (r2022)') %>%
   select(MODEL,SCENARIO,ISO,VARIABLE,UNIT,RYEAR,EJ) %>% 
   spread(RYEAR, EJ) -> histData.eleGen


 write.xlsx(histData.pe, paste("IEA_history_IPCCSR15-ISO-2021-EB-subset.xlsx", sep = ''), sheetName = 'data_PE', append = F, row.names = F)
 write.xlsx(histData.eleGen, paste("IEA_history_IPCCSR15-ISO-2021-EB-subset.xlsx", sep = ''), sheetName = 'data_ElGen', append = T, row.names = F)

}

