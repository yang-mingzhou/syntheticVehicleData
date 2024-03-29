# Synthetic_Dataset_Generation

## Installation and Development

A. Install OSMnx
* Install osmnx following https://osmnx.readthedocs.io/en/stable/installation.html
(```conda create -n syntheticData -c conda-forge --strict-channel-priority python=3.8.17 osmnx```)

B. Install FASTsim and SUMO within the environment
* Install fastsim following https://github.com/NREL/fastsim (```pip install fastsim```, note that FASTSim is compatible with Python 3.8 - 3.10)
* Install sumo following https://sumo.dlr.de/docs/Installing/index.html (```pip install eclipse-sumo==1.18.0``` and then ```pip install traci```)
* Install mappymatch https://github.com/NREL/mappymatch (```pip install mappymatch```)

* Any other libraries and their versions are listed in environment.yml

C. Execution:

1) Activate the conda env: ```source activate syntheticData```

2) Generate synthetic data from read data: ```python syntheticDataGenFromRealData.py```

or 

3) Generate synthetic data randomly (the trips are generated from shortest paths between random origin-destination pairs):
```python randomSyntheticDataGen.py```


