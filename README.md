# Assessing Vulnerability for use in Determining Basin-scale Environmental Watering Priorities

## Executive summary 
The planning and prioritisation of environmental water is a key step to achieving the long-term environmental outcomes of the Basin-wide environmental watering strategy (BWS). The method for determining Basin annual environmental watering priorities is constantly improving and aims to take into account a wide range of factors including vulnerability of ecosystems and the biota that depend on them. The availability of Basin-wide data, particularly from remote sensing has increased dramatically in the past few years and represents an opportunity to develop and implement a systematic approach to identifying vulnerable ecosystems and biota. This project developed and tested a method for assessing vulnerability for two of the four BWS themes, native vegetation, and waterbirds.

The method is underpinned by a logic, consistent with other vulnerability assessments, where by vulnerability is considered to be a product of condition (how sensitive biota are to withstand environmental change and their ability to adjust to those changes) and stress (exposure to adverse environmental changes). Spatial indicators of condition and stress were based on our conceptual understanding of inundation dependent vegetation communities and waterbirds. Conceptual models of the factors that affect vulnerability were developed for each theme and guided the selection of indicators. While initially a long list of potential indicators was developed for each theme, the availability of suitable data at a Basin-scale limited the final selection. The final list comprised three indicators of condition for vegetation (tree stand condition, vegetation cover and “greenness”) with three indicators of stress (time since last inundation, inundation extent and soil root zone moisture) and four indicators of condition for waterbirds (abundance, species richness, breeding abundance, breeding species richness) with four indictors of stress (extent of inundation, time since last inundation, rainfall, “greenness” of vegetation).

Indicators are not scored absolutely but assigned to rank categories from better to worse condition and from low to high stress. Two different methods were used for assigning indicators into ranks. For most indicators, a change from baseline conditions has been used. For a small number of indicators, absolute thresholds based on known species/functional group tolerances have been established.
The results of the condition, stress and vulnerability assessment can be presented in two ways: spatially as a map or temporally in a table. The spatial presentation allows visual comparisons of different areas in the Basin at a single point in time. The temporal presentation of results allows visual comparison of a single spatial unit across multiple years.

The method appears to provide a robust way of assessing condition, stress and vulnerability at large spatial scales despite data limitations, uncertainties and the assumptions that underpin the method. The comparisons with the Millennium Drought (where there is empirical evidence of a decline in condition and increase in stress and vulnerability) revealed expected patterns with high vulnerability suggesting the method is sensitive to revealing patterns of vulnerability to water stress that can inform management. It must be recognised, however, there will always be better, finer-scale information to inform watering requirements at the site and local scale.

Priorities for environmental water will require consideration of a variety of factors such as cultural value, feasibility, watering history and competing priorities. The vulnerability assessment as described here, can provide a valuable input to the prioritisation process for environmental water.

A series of Jupyter notebooks, published within this repository, have been developed that contain the method that would allow for annual assessment of vulnerability for waterbirds and native vegetation. The data associated with this project are avaliable on request from the MDBA at https://www.mdba.gov.au/publications/maps-spatial-data/geospatial-data-services-request-form.

For more information see: Commonwealth of Australia 2023, _Assessing Vulnerability for use in Determining Basin-scale Environmental Watering Priorities_, Canberra. CC BY 4.0.
https://www.dcceew.gov.au/sites/default/files/documents/assessing-vulnerability-use-determining-basin-scale-environmental-watering-priorities.pdf

# Included in this repository
* BWSVulnerability Python Environment.txt - details the python packages and version numbers needed to run the project code
* MDBA Tree Stand Condition Tool Rasters.txt - Google Earth Engine code to extract Stand condition tool rasters
* Normalized Difference Vegetation Index.txt - Google Earth Engine code to extract Normalized Difference Vegetation Index rasters 
* bws_vulnerability.ipynb - Main python notebook for final data processing to produce outputs
* wit_metrics.ipynb - Python notebook for calculating metrics on the Wetland insight tool data
* workers.py - Python script of metric calculation function setup for multiprocessing


