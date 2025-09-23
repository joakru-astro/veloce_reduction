# veloce_reduction

Python code for simple reduction of data from Veloce @ AAT.

For more information than the overview below, check the [Wiki](https://github.com/joakru-astro/veloce_reduction/wiki).

### Statement of purpose
[Veloce](https://aat.anu.edu.au/science/instruments/current/veloce/overview) is a high-resolution ($\lambda / \Delta \lambda = 80'000$) échelle spectrograph, covering the wavelength range from 396nm to 940nm in a single exposure. The red arm was commissioned in 2018 [(Gilbert et al. 2018)](https://ui.adsabs.harvard.edu/abs/2018SPIE10702E..0YG/abstract), with green and blue joining in 2023 [(Taylor et al. 2024)](https://ui.adsabs.harvard.edu/abs/2024SPIE13096E..45T/abstract). However, despite the community being granted access to the instrument and being open to observing proposals, there was no official public pipeline for data reduction. Having the opportunity to observe with Veloce@AAT and spend hours working with the data, I decided to share my code to process the observations.

### Workflow
The reduction workflow consists of the following steps:

1. Load and verify config
2. Read logs and list of science targets
3. Load traces
4. \[Optional\] If using static wavelength calibration load precomputed solution
5. \[Optional\] Compute master flat
6. Extract science data:
  - remove bias (using overscan)
  - \[Optional\] divide by master flat
  - \[Optional\] remove scattered light
  - extract 1D spectrum by summing flux in cross-dispersion direction within trace region
  - assign wavelengths to pixels (static solution or simultaneous LC)
  - convert vacuum wavelength to air
  - save fits file

### Contents
The repository consists of the following: 
```
veloce_reduction/
├── README.md                           # This file
├── LICENSE                             # License file ###TODO
├── config.yaml                         # Example configuration file for the reduction
├── veloce_reduction.py                 # The script running reduction based on the configuration file
├── veloce_reduction.ipynb              # As above but notebook
├── step-by-step_extraction.ipynb       # Example step by step 1d spectrum extraction
│
├── simple_veloce_reduction/            # A python module handling the reduction
│   ├── __init__.py                     # 
│   ├── veloce_config.py                # Configuration handling and path management
│   ├── veloce_extraction.py            # High-level functions for data extraction
│   ├── veloce_reduction_tools.py       # Low-level reduction utilities
│   ├── veloce_wavecalib.py             # Wavelength calibration functions
│   ├── veloce_diagnostic.py            # Diagnostic plotting functions
│   ├── Trace/                          # Trace templates (custom and ones from Veloce Manual)
│   │   └── ...
│   └── Wave/                           # Wavelength calibration data (custom and ones from Veloce Manual)
│       └── ...
└── [additional content]                # Notebooks with in detail investigations some wip
```
### Requirements

Python distribution

Install the following non-standard Python packages:

```bash
pip install astropy csaps matplotlib numpy scipy PyYAML scikit-learn
```

Or using conda:
```bash
conda install astropy matplotlib numpy scipy PyYAML scikit-learn
conda install -c conda-forge csaps
```

**Tested with:**
Python version: 3.12.1 

Package versions:
astropy: 5.3.4
numpy: 1.26.3
scipy: 1.11.4
matplotlib: 3.8.0
scikit-learn: 1.7.1
PyYAML: 6.0.1
csaps: 1.1.0

### Limitations
In its current form, my pipeline doesn't use optimal extraction and just co-adds the flux from all the fibres; thus, it is probably not suited for precise RV science.

### Data
Place your data in the /Data/Raw/ or specify the path to the custom location in config.yaml.
[Sample data from CSV run](https://www.dropbox.com/scl/fo/qleydw5tsvpkpfl1jx985/AJmCFf8FNwuO7GEVXNzZSDQ?rlkey=xy9j0vfnb0wo6xzcf27n6wgpu&st=4gq4srkk&dl=0)

### Acknowledgement
For the static wavelength solution, I use precomputed coefficients for polynomials available in the Veloce Instrument Manual. I want to thank Chris Tinney for these solutions and all the valuable information provided in the manual, and I extend my gratitude to anyone who contributed to this resource.
