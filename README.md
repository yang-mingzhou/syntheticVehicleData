# Synthetic_Dataset_Generation

## Installation and Development

A. Install OSMnx
* Install osmnx following https://osmnx.readthedocs.io/en/stable/installation.html
conda create -n envNameExample -c conda-forge --strict-channel-priority python=3.8.17 osmnx
* Any other libraries and their versions are listed in osmnx/requirements.txt


B. Install FASTsim and SUMO
* Install fastsim following https://github.com/NREL/fastsim (```pip install fastsim```, note that FASTSim is compatible with Python 3.8 - 3.10)
* Install sumo following https://sumo.dlr.de/docs/Installing/index.html (for python package install: ```pip install eclipse-sumo==1.18.0``` and then ```pip install traci```)
* Install mappymatch https://github.com/NREL/mappymatch (```pip install mappymatch```)

* Any other libraries and their versions are listed in sumo_fastim/requirements.txt

C. ```source activate syntheticData```
```python syntheticDataGenFromRealData.py```
```python randomSyntheticDataGen.py```


