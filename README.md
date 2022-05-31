# BGI ML Pipeline

Contains two separate pipelines which take simulated data generated via VIPM, or real data from NXCALS, and create models to predict undistorted beam parameters.

## Getting started

## NXCALS Pipeline
Having taken BGI data it will be stored in NXCALS. The NXCALS pipeline will access this data and restructure it to required format for training. It will then convert to pandas and train. It shows 3 example models and the scatter plots of 'predicted' vs 'real' beam width as well as the correlations of these two variables. 
The NXCALS is the most automated of both pipelines but requires 3 user changes:
1. Input the correct dates and times of your data taking. In theory you can very liberally bracket your data taking period and it should drop anything that doesn't have real data but I saw some issues with this, possibly due to our operation of the BGI and possibly due to some unknown error. In any case, the closer you can bracket your data the better. 
2. SPLIT_COUNT i.e.: how many profiles did you take per cycle (90)
3. SPLIT_SIZE i.e.: how many bins is each profile binned into (1024)
These two variables are required because NXCALS stores the profile data as a 1D array of length (SPLIT_COUNT*SPLIT_SIZE) and you will need to input how it is to be split. Again, in theory this data is stored as a variable and can be used row by row for extracting the profiles. The lines where this data would be pulled are commented out in the pipeline.
All operations on the df are pyspark operations which is based on SQL database operations. For information on how these work see: https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-pyspark-rdd-operations/ 

## VIPM Pipeline
Having run a VIPM parameter sweep you should have a list of result files stored somewhere in EOS. This pipeline will take that file location. Read each result.csv in turn and store the data in a pandas dataframe for use throughout. It will then train various different models and test their accuracy. Use this to see how certain things effect your models (ie what is the effect of many different currents on model) not to train for a model that will work on real data. The models trained here are unlikely to work on real data due to the differences that cannot be simulated. In theory if the beam shapes looked the same there is potential it could work. 
The user changes required here are:
1. fileloc = eos folder where your results are stored
2. bin_width = must be set at the same value you used in the data extraction. 
This notebook relies heavily on the modelUtils python library which is admittedly poorly commented but should be simple enough. 

Simulation data is stored on EOS at: /eos/project/b/bgi/PS/simulation/ml_simulation_data

## Installation
No installation required for either pipeline. Just NXCALS access and a browser to open CERN SWAN. However, to run VIPM you need a Python environment with VIMP-GUI and all dependencies installed. See: https://ipmsim.gitlab.io/Virtual-IPM/index.html for this. 

## Usage
The initial commit of these files contains examples in the form of a completed notebook, just either browse through the initial commit, or pull it and run in a container. 

## Support
Contact lgolino@cern.ch for help.

## Useful Resources
FESAweb link to see profile acquisition in real time: https://fesaweb.cern.ch/ondemand?device=PR.BGI82&property=ProfileAcquisition&fields=profileTotalNumberOfEvents,profiles&cycle=CPS.USER.TOF

Timber config to pull or view data: https://timber.cern.ch/query?tab=Visualization&configuration=36742

How to create NXCALS Swan session: https://nxcals-docs.web.cern.ch/current/user-guide/swan/swan-session/ 

## To Do
### NXCALS:
1. Make the NXCALS pipeline use built in spark ML libraries instead of converting to pandas and using standard ML libraries. This is untested and currently not required but with more and more data this will soon become necessary.
2. Expand NXCALS pipeline to attempt to reconstruct full profile instead of beam width.
3. Make NXCALS pipeline use the given dimensions for each row when extracting profiles. 
4. Have it pull the bin locations from profileTransversePositions_um instead of just a linear 56mm array.
### VIPM:
1. Standardise data import
2. Generally clean the pipeline. It works but it's not the cleanest solution in the sense that it still requires a lot of human tuning and manipulation to work.
3. There is an issue where setting the "drop_params" as False will create a model incompatible with the cells in the comparing differences in datasets. One is 400 and the other 402. This needs to be standardised. 


## Project status
Contract with BI group ended, I would like to add more to this and really complete it but it's time dependent.
