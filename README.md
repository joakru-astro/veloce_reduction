# veloce_reduction

Python code for simple reduction of data from Veloce @ AAT.

### Workflow
The reduction workflow consists of follwoing steps:

(data extraction)

1. Read logs
2. Save targets list
3. Extract science data

(merging and normalisation)

5. (optional) Make master flat
6. Extract the blaze function from master flat
7. Co-add orders
8. Divide co-added science spectrum by co-added blaze
9. Renormalise merged spectrum (this step is using SUPPNet)

### Contents
Repository consists of the following: 
- veloce_reduction.ipynb - notebook presenting minimal working example (steps 1-3 as of now)
- simple_veloce_reduction - a python module handling the reduction including:
  - veloce_reduction_tools.py - low level functions used in reduction
  - veloce_path.py - class hendling the paths, can be edited for custom localisations of components
  - veloce_logs.py - functions used to read standard veloce logs and saving targets list
  - veloce_extraction.py - high level functions for data extraction
- \[deprecated\] veloce_trace.ipynb - notebook containing extraction of trace based on flat field image
- \[deprecated\] veloce_wave_calib.ipynb - notebook checking precomputed wave calibration
- \[deprecated\] extract_one_night.ipynb - old notebook for extraction data from one night based on targets list

### Requierments
Conda environment can be recreated using: 
```
conda env create -f environment.yml
```
I aim to clean this environment and make a full instalation instruction. And probably rename the environment.

### Data
[Sample data from CSV run](https://www.dropbox.com/scl/fo/qleydw5tsvpkpfl1jx985/AJmCFf8FNwuO7GEVXNzZSDQ?rlkey=xy9j0vfnb0wo6xzcf27n6wgpu&st=4gq4srkk&dl=0)
