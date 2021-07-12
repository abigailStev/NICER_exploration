# NICER_exploration

Abigail Stevens, abigailstev@gmail.com, copyright 2018-2021
Licensed with an MIT license

## Order of operations
### 1. query_dl_nicer.ipynb

Searches the NASA server for data files on your source, writes scripts and uses WGET to download the files to your harddrive.

### 2. make_GTIs.ipynb

Opens the event files, checks the built-in GTIs in the second FITS HDU, only keeps the GTIs longer than some set length (like 16 seconds, so that you don't have a bunch of 1-second GTIs hanging around), saves a table of the good longer GTIs to a fits table, and saves the list of those tables to a file that should correspond line-by-line with the list of event files.

### 3. quicklook_segments.py

Using the event lists and GTI lists, it steps through the data files computing things like the count rate, hardness ratio, and rms per short segment (like 16 seconds) and per GTI.

### 4. segment_info.ipynb

Notebook to look through the output quicklook tables from step 3 to make pretty plots and figure out exactly which files you want to do further analysis on.

### 5. dyn_psds.py

Makes dynamical power spectra.

### 6. just_plotting_dyn_psds.py

Takes the output from step 5 and just tweaks the plot, so that you can edit your axis limits without needing to recompute the dynamical power spectra all over again.


## Other stuff
### ci_bins.ipynb
For calculating what the edges of the bin ranges should be for the FTOOLS commands RBNPHA and RBNRMF.

### maxi_plot.ipynb
Makes plots of photon count and hardness vs days from MAXI data (need to download MAXI data by hand).

### xcor_tools_nicer.py
Helper classes and methods that are imported and used in all of the above software.


## Depreciated
Use tools from Stingray Software instead of these old programs: lag-energy.ipynb, lag-frequency.ipynb, power_and_cross.py, rms-energy_spectra.ipynb
