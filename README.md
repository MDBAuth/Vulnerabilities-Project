# Assessing Vulnerability for use in Determining Basin-scale Environmental Watering Priorities

## Executive summary 
The planning and prioritisation of environmental water is a key step to achieving the long-term environmental outcomes of the Basin-wide environmental watering strategy. The method for determining Basin annual environmental watering priorities is constantly improving and aims to take into account a wide range of factors including vulnerability of ecosystems and the biota that depend on them. The availability of Basin-wide data, particularly from remote sensing has increased dramatically in the past few years and represents an opportunity to develop and implement a systematic approach to identifying vulnerable ecosystems and biota. This project developed and tested a method for assessing vulnerability for two of the four BWS themes, native vegetation, and waterbirds. The specific objectives of the project were to:

*	Develop a GIS based method for assessing Basin-scale vulnerability for two thematic areas of the Basin-wide environmental watering strategy (native vegetation and waterbirds). The method must be in a format that is repeatable and can be routinely updated by MDBA agency staff as part of business-as-usual operations.
*	Producing outputs for the selected themes using applicable GIS/spatial analysis tools:
  *	indicators of stress and condition for each of the two themes
  *	an overall basin-scale assessment of vulnerability for each theme
  *	vulnerability of vegetation and waterbirds at different spatial scales applicable to water management.
*	Consideration of confidence in the source data and outputs.

The method is underpinned by a logic, consistent with other vulnerability assessments, where by vulnerability is considered to be a product of condition (how sensitive biota are to withstand environmental change and their ability to adjust to those changes) and stress (exposure to adverse environmental changes).
Spatial indicators of condition and stress were based on our conceptual understanding of inundation dependent vegetation communities and waterbirds. Conceptual models of the factors that affect vulnerability were developed for each theme and guided the selection of indicators. While initially a long list of potential indicators was developed for each theme, the availability of suitable data at a Basin-scale limited the final selection. The final list comprised three indicators of condition for vegetation (tree stand condition, vegetation cover and “greenness”) with three indicators of stress (time since last inundation, inundation extent and soil root zone moisture) and four indicators of condition for waterbirds (abundance, species richness, breeding abundance, breeding species richness) with four indictors of stress (extent of inundation, time since last inundation, rainfall, “greenness” of vegetation).
Indicators are not scored absolutely but assigned to rank categories from better to worse condition and from low to high stress. Two different methods were used for assigning variable into ranks. For most indicators, a change from baseline conditions has been used. For a small number of indicators, absolute thresholds based on known species / function group tolerances have been established.
The results of the condition, stress and vulnerability assessment can be presented in two ways: spatially as a map or temporally in a table. The spatial presentation allows visual comparisons of different areas in the Basin at a single point in time. The temporal presentation of results allows visual comparison of a single spatial unit across multiple years.
A series of Jupyter notebooks has been developed that contains the method and would allow for annual assessment of vulnerability for waterbirds and native vegetation with available input data.

Hale, J., Brooks, S., Campbell, C. and McGinness, H. (2023) Assessing Vulnerability for use in Determining Basin-scale Environmental Watering Priorities. A Report to the Commonwealth Environmental Water Office, Canberra.

# Included in this repository
* BWSVulnerability Python Environment.txt - details the python packages and version numbers needed to run the project code
* MDBA Tree Stand Condition Tool Rasters.txt - Google Earth Engine code to extract Stand condition tool rasters
* Normalized Difference Vegetation Index.txt - Google Earth Engine code to extract Normalized Difference Vegetation Index rasters 
* bws_vulnerability.ipynb - Main python notebook for final data processing to produce outputs
* wit_metrics.ipynb - Python notebook for calculating metrics on the Wetland insight tool data
* workers.py - Python script of metric calculation function setup for multiprocessing


