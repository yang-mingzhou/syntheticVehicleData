import pandas as pd
import matplotlib.pyplot as plt
import ast
import matplotlib.dates as mdates
import sumolib
import osmnx as ox

import sys
import os
import glob
import random
import logging
import subprocess
import numpy as np

from fastsim import simdrive, vehicle, cycle
from fastsim import parameters as params

import traci
import time
import csv

import xml.etree.ElementTree as ET
import xml.dom.minidom

from datetime import datetime, timedelta

from utils.simulation import simulation 
from utils.featureExtraction import feature_extraction
# postprocessing to edge level
from utils.postprocessing import postprocessing


data_file_folder = "data"
results_file_folder = "results"

# Set SUMO_HOME; revise it according to the path to the site-packages folder of SUMO  
os.environ['PATH'] += ":/home/shekhars/yang7492/.conda/envs/syntheticData/lib/python3.8/site-packages/sumo/bin"
os.environ['SUMO_HOME'] = '/home/shekhars/yang7492/.conda/envs/syntheticData/lib/python3.8/site-packages/sumo'

# simulataion and save
input_file_pattern = os.path.join(data_file_folder, "matchedTrips/Murphy/*_matched.csv")
simulation_output_folder = os.path.join(data_file_folder, "simulated/Murphy") 

simulation(data_file_folder, results_file_folder, input_file_pattern, simulation_output_folder)

# conver the trip level data to edge level and save
edge_level_output_folder = os.path.join(data_file_folder, "processed/Murphy") 
postprocessing(data_file_folder, results_file_folder, simulation_output_folder, edge_level_output_folder)


# extract features for neural network training
edge_level_file_pattern = os.path.join(edge_level_output_folder, "*_processed.csv")
output_file_folder =  os.path.join(data_file_folder, "features/Murphy")
# process synthetic data or real-world data 
synthetic_data_flg = True # or False for processing real-world data
feature_extraction(data_file_folder, results_file_folder, edge_level_file_pattern, output_file_folder, synthetic_data_flg)