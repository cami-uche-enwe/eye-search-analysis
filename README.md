# eye-search-analysis
Scripts for parsing raw eye data, aligning with raw behavioural data, and data cleaning/plotting

*parse_eye_align_rawdata.ipynb* is a Jupyter notebook where you:
1. import raw eye data in .asc format
2. import raw behavioural data in .csv format
3. parse through eye data to extract all measures of interest for each trial (e.g., n of fixations on the target, latency of saccades, etc)
4. align parsed eye data with behavioural data
5. carry out data cleaning
6. plot some dependent variables for sanity checks
7. pivot data for dependent variables of interest to export for JASP analysis
8. prepare data to suit SMART expected format

*parse_eye_align_rawdata.ipynb* requires some functions in *myutils.py* and *pygazeanalyser* 


*runSMART.ipynb* is a Jupyter notebook where you:
1. run SMART analysis on salience effect pre and post interruption
2. ...

All files needed for running *runSMART.ipynb* are in the SMART dir.

All files are still under development as of November 2025.