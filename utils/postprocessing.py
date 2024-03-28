import pandas as pd
import matplotlib.pyplot as plt
import ast
import matplotlib.dates as mdates
import sumolib
import osmnx as ox

import sys
import os
import math
import glob
import json
import requests
import numpy as np

import traci
import time
import csv

from fastsim import simdrive, vehicle, cycle
from fastsim import parameters as params

from mappymatch.constructs.geofence import Geofence
from mappymatch.utils.plot import plot_geofence
from mappymatch.maps.nx.nx_map import NxMap, NetworkType

from itertools import chain
from time import sleep
from datetime import timedelta

import xml.etree.ElementTree as ET
import xml.dom.minidom

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from utils.postprocessingUtils import trip_to_edge_df, cal_lanes, cal_bridge, edge_list_to_node_list



def read_data(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Remove rows with any empty values
    df = df.dropna()
    
    if not len(df):
        return df
    
    # Convert string representations of lists or tuples back to lists or tuples
#     columns_to_eval = ['altitude_profile', 'velocity_profile', 'weight', 'total_fuel', 'trajectory', 
#                        'matched_path', 'coordinate_id', 'road_id']
    columns_to_eval = ['velocity_profile', 'weight', 'total_fuel', 'trajectory', 
                        'matched_path', 'coordinate_id', 'road_id']
    for column in columns_to_eval:
        df[column] = df[column].apply(ast.literal_eval)

    # Remove fractional seconds and convert 'time' to datetime
    time_columns = ['trip_start_time', 'trip_end_time']
    for column in time_columns:
        df[column] = df[column].str.split('.').str[0]
        df[column] = pd.to_datetime(df[column], format='%Y-%m-%d %H:%M:%S')

    # Convert space-separated strings to lists of floats
    float_list_columns = ['fastsim_velocity', 'fastsim_power', 'sumo_velocity']    
    
    for column in float_list_columns:
        df[column] = df[column].apply(lambda x: [float(i) for i in x.split()])

    # Convert space-separated strings to lists
    df['sumo_path'] = df['sumo_path'].str.split()

    return df




def extracted_matched_edges(row):
    starting_index = row['coordinate_id'][0]
    row["trip_start_time"] = pd.to_datetime(row['trip_start_time']) + pd.to_timedelta(starting_index, unit='s')
#     row['altitude_profile'] = row['altitude_profile'][starting_index:]
    row['velocity_profile'] = row['velocity_profile'][starting_index:]
    row['weight'] = [row['weight'][0]] * len(row['velocity_profile'][starting_index:])
    row['total_fuel'] = row['total_fuel'][starting_index:]
    row['trajectory'] = row['trajectory'][starting_index:]
    row['road_id'] = row['road_id'][starting_index:]
    return row

def fetch_elevations_batch(locations, retries=3, delay=5):
    url = "https://api.open-elevation.com/api/v1/lookup"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    for attempt in range(retries):
        response = requests.post(url, headers=headers, data=json.dumps({"locations": locations}))
        if response.status_code == 200:
            return response.json()['results']
        else:
            print(f"Attempt {attempt+1} failed with status code {response.status_code}. Retrying after {delay} seconds...")
            sleep(delay)
    raise Exception(f"API Request failed after {retries} attempts, status code {response.status_code}, response: {response.text}")

def add_elevation_change_to_edges(nodes, edges):
    # Extract locations from nodes for elevation fetching
    locations = [{"latitude": row['y'], "longitude": row['x']} for idx, row in nodes.iterrows()]
    
    # Fetch elevations in batches and create a dictionary mapping node ID to elevation
    batch_size = 10000  # Adjust based on API limits
    all_elevations = []
    total_batches = (len(locations) + batch_size - 1) // batch_size  # Calculate the total number of batches
    print(f"Elevation total batches to fetch : {total_batches}")
    
    for i in range(0, len(locations), batch_size):
        print(f"Elevation processing batch {i // batch_size + 1} of {total_batches}")
        batch = locations[i:i + batch_size]
        batch_results = fetch_elevations_batch(batch)
        all_elevations.extend(batch_results)
    
    node_elevation = {node: elevation for node, elevation in zip(nodes.index, [res['elevation'] for res in all_elevations])}

    # Calculate elevation change for each edge
    def calculate_elevation_change(row):
        u, v, _ = row.name  # row.name contains the index of the row, which in this case is (u, v, key) for the edge
        start_elev = node_elevation[u]
        end_elev = node_elevation[v]
        return end_elev - start_elev

    # Apply the elevation change calculation
    edges['elevation_change'] = edges.apply(calculate_elevation_change, axis=1)
    
    return edges



    
def process_edge_gdf(edge_gdf, node_gdf, results_file_folder):
    # lanes
    edge_gdf['lanes_normed'] = edge_gdf['lanes'].apply(cal_lanes)
    edge_gdf['lanes_normed'] = edge_gdf['lanes_normed'].apply(lambda x: x if x <= 8 else 8)
    
    # bridges
    edge_gdf['bridge_normed'] = edge_gdf['bridge'].apply(lambda x: cal_bridge(x))
    
    # endpoints features
    edge_gdf['u'] = edge_gdf.index.get_level_values(0)
    edge_gdf['v'] = edge_gdf.index.get_level_values(1)
    edge_gdf['key'] = edge_gdf.index.get_level_values(2)
    edge_gdf['signal_u'] = edge_gdf.u.apply(lambda x: node_gdf.loc[x, 'highway']).fillna("None")
    edge_gdf['signal_v'] = edge_gdf.v.apply(lambda x: node_gdf.loc[x, 'highway']).fillna("None")
    list_uv = list(edge_gdf.signal_u.unique())+list(edge_gdf.signal_v.unique())
    endpoints_dictionary = dict()
    endpoints_dictionary['None'] = 0
    cnt_endpoint = 1
    for i in list_uv:
        if i not in endpoints_dictionary:
            endpoints_dictionary[i] = cnt_endpoint
            cnt_endpoint += 1
    np.save(os.path.join(results_file_folder, 'endpoints_dictionary.npy'), endpoints_dictionary)
    edge_gdf['signal_u_value'] = edge_gdf.signal_u.apply(lambda x: endpoints_dictionary[x])
    edge_gdf['signal_v_value'] = edge_gdf.signal_v.apply(lambda x: endpoints_dictionary[x])
    
    # add edge_gdf['elevation_change']
    edge_gdf = add_elevation_change_to_edges(node_gdf, edge_gdf)
    
    return edge_gdf


def process_one_file(processed_df, edge_gdf, sumo_net):
    processed_df['sumo_node_path'] = processed_df['sumo_path'].apply(lambda x: edge_list_to_node_list(x, sumo_net))
    processed_df = processed_df.apply(extracted_matched_edges, axis=1)
    df_edges = trip_to_edge_df(processed_df, edge_gdf)
    print(len(df_edges))
    return df_edges

    

def postprocessing(data_file_folder, results_file_folder, input_folder, output_folder):
    # load SUMO map
    net_file = os.path.join(data_file_folder, 'Minneapolis.net.xml')
    sumo_net = sumolib.net.readNet(net_file)

    # load openstreet map:
    osm_file_path = os.path.join(data_file_folder, "maps/minneapolis.graphml")
    osmnx_net = ox.io.load_graphml(osm_file_path)
    node_gdf, edge_gdf = ox.utils_graph.graph_to_gdfs(osmnx_net)
    edge_gdf = process_edge_gdf(edge_gdf, node_gdf, results_file_folder)
        
    # Define the input and output directories
    data_folder = input_folder

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Pattern to match all CSV files in the data folder
    pattern = os.path.join(data_folder, "*.csv")

    # Iterate over all CSV files in the data folder
    for file_name in glob.glob(pattern):
        print(f"Processing {file_name}...")
        
        # Generate output file name based on input file
        base_name = os.path.basename(file_name)
        
        output_file_name = os.path.join(output_folder, f"edge_level_{base_name}")
        
        # Check if the processed file already exists
        if os.path.exists(output_file_name):
            continue  # Skip processing this file and move to the next one

        # Assuming read_data, edge_list_to_node_list, extracted_matched_edges, trip_to_edge_df are defined
        processed_df = read_data(file_name)
        
        if not len(processed_df):
            continue
               
        df_edges = process_one_file(processed_df, edge_gdf, sumo_net)
        
        # Save the processed DataFrame to a new CSV in the output folder
        df_edges.to_csv(output_file_name, index=False)

        print(f"Saved processed data to {output_file_name}")
        


if __name__ == "__main__":
    os.environ['PATH'] += ":/home/shekhars/yang7492/.conda/envs/syntheticData/lib/python3.8/site-packages/sumo/bin"
    # Set SUMO_HOME
    os.environ['SUMO_HOME'] = '/home/shekhars/yang7492/.conda/envs/syntheticData/lib/python3.8/site-packages/sumo'
    data_file_folder = "../data" 
    results_file_folder = "../results"
    input_folder = os.path.join(data_file_folder, "synthetic/Murphy")
    output_folder = os.path.join(data_file_folder, "processed/Murphy")  # Make sure this exists or create it
    postprocessing(data_file_folder, results_file_folder, input_folder, output_folder)