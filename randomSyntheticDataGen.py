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

import utils.simulation as sim 
from utils.featureExtraction import feature_extraction
# postprocessing to edge level
from utils.postprocessing import process_edge_gdf, process_one_file, edge_list_to_node_list

number_of_trips = 1000

data_file_folder = "data"
results_file_folder = "results"

edge_level_data_folder = os.path.join(data_file_folder, "processed/synthetic")
results_data_file_folder = os.path.join(data_file_folder, "features/synthetic")

os.makedirs(edge_level_data_folder, exist_ok=True)  # Create the destination folder if it doesn't exist
os.makedirs(results_data_file_folder, exist_ok=True) 

# Set SUMO_HOME; revise it according to the path to the site-packages folder of SUMO  
os.environ['PATH'] += ":/home/shekhars/yang7492/.conda/envs/syntheticData/lib/python3.8/site-packages/sumo/bin"
os.environ['SUMO_HOME'] = '/home/shekhars/yang7492/.conda/envs/syntheticData/lib/python3.8/site-packages/sumo'

# load openstreet map:
osm_file_path = os.path.join(data_file_folder, "maps/minneapolis.graphml")
osmnx_net = ox.io.load_graphml(osm_file_path)
node_gdf, edge_gdf = ox.utils_graph.graph_to_gdfs(osmnx_net)

# load sumo map:
net_file = os.path.join(data_file_folder, "Minneapolis.net.xml")
sumo_net = sumolib.net.readNet(net_file)

# SUMO simulation configuration
# file name for random O-D pairs
od_file = os.path.join(data_file_folder, "incompelete_routes.xml")

# file name for complete routes
complete_route_file = os.path.join(data_file_folder, "complete_routes.rou.xml")

# generate one trip every second
sim.construct_complete_route(os.environ['SUMO_HOME'], number_of_trips, number_of_trips, net_file, od_file, complete_route_file)

sim.add_speed_attributes_to_vehicles(complete_route_file, "0.00", "0.00")


vehicle_ids = [f"t{i}" for i in range(number_of_trips)]

begin = 0
end = 14400 # maximum simulated travel time 
step_length = 1
file_name_config = "sumo.sumocfg"
sim.save_sumo_config_to_file(net_file, complete_route_file, begin, end, step_length, file_name_config)

# sumo simulation
velocity_data, edgeSeq_data = sim.sumo_simulation(file_name_config, vehicle_ids)

# Generate synthetic vehicle type
# 62 predefined vehicle types in FASTSim: https://github.com/NREL/fastsim/blob/fastsim-2/python/fastsim/resources/FASTSim_py_veh_db.csv
# vehicle_type = np.random.randint(1, 27, number_of_trips)
# vehicle_type = 0 -> the predefined Murphy Heavey Duty Truck. Refer to cumstomize_veh() in simultation.py
vehicle_type = [0 for _ in range(number_of_trips)]

# generte a dataframe of the synthetic data based on SUMO results
csv_file = sim.generate_synthetic_csv(velocity_data, edgeSeq_data, vehicle_type, edge_gdf, sumo_net)

# Process the data (trip level) and save to the destination folder
simulated_data = sim.fastsim(csv_file, velocity_data, edgeSeq_data, data_file_folder)

# Convert space-separated strings to lists of floats or handle already-converted floats
float_list_columns = ['fastsim_velocity', 'fastsim_power', 'sumo_velocity']
for column in float_list_columns:
    simulated_data[column] = simulated_data[column].apply(lambda x: [float(i) for i in x.split()] if isinstance(x, str) else x)

# Convert space-separated strings to lists
simulated_data['sumo_path'] = simulated_data['sumo_path'].str.split()


# conver the processed data to edge level
edge_gdf = process_edge_gdf(edge_gdf, node_gdf, results_file_folder)
df_edge = process_one_file(simulated_data, edge_gdf, sumo_net)
# Save the processed DataFrame to a new CSV in the output folder
edge_level_output_file_name = os.path.join(edge_level_data_folder, "synthetic.csv")
df_edge.to_csv(edge_level_output_file_name, index=False)
print(f"Saved processed data to {edge_level_output_file_name}")

edge_level_file_pattern = os.path.join(edge_level_data_folder, "*.csv")
synthetic_data_flg = True
feature_extraction(data_file_folder, results_file_folder, edge_level_file_pattern, results_data_file_folder, synthetic_data_flg)
print(f"Saved extracted feature data to {results_data_file_folder}")