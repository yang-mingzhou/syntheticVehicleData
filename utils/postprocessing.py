import pandas as pd
import matplotlib.pyplot as plt
import ast
from datetime import timedelta
import matplotlib.dates as mdates
import sumolib
import osmnx as ox

import sys
import os
import math
import glob
import numpy as np

from fastsim import simdrive, vehicle, cycle
from fastsim import parameters as params

from mappymatch.constructs.geofence import Geofence
from mappymatch.utils.plot import plot_geofence
from mappymatch.maps.nx.nx_map import NxMap, NetworkType

import traci
import time
import csv

import xml.etree.ElementTree as ET
import xml.dom.minidom

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from postprocessingUtils import trip_to_edge_df, cal_lanes, cal_bridge


def read_data(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Remove rows with any empty values
    df = df.dropna()
    
    # Convert string representations of lists or tuples back to lists or tuples
    columns_to_eval = ['altitude_profile', 'velocity_profile', 'weight', 'total_fuel', 'trajectory', 
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


# Function to convert edge list to node list
def edge_list_to_node_list(edges, sumo_net):
    # Split the edge list string into individual edges
    node_list = []
    for edge in edges:
        if edge.startswith(':'):
            node_list.append(node_list[-1])
        else:
            edge_obj = sumo_net.getEdge(edge)  # Get the edge object
            from_node_id = edge_obj.getFromNode().getID()  # Get the ID of the origin node
            to_node_id = edge_obj.getToNode().getID()  # Get the ID of the destination node
            node_tuple = (from_node_id, to_node_id, 0)  # Create a tuple 
            node_list.append(node_tuple)
    return node_list

def extracted_matched_edges(row):
    starting_index = row['coordinate_id'][0]
    row["trip_start_time"] = pd.to_datetime(row['trip_start_time']) + pd.to_timedelta(starting_index, unit='s')
    row['altitude_profile'] = row['altitude_profile'][starting_index:]
    row['velocity_profile'] = row['velocity_profile'][starting_index:]
    row['weight'] = [row['weight'][0]] * len(row['weight'][starting_index:])
    row['total_fuel'] = row['total_fuel'][starting_index:]
    row['trajectory'] = row['trajectory'][starting_index:]
    row['road_id'] = row['road_id'][starting_index:]
    return row
    
def preprocess_edge(edge_gdf, node_gdf):
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
    np.save('../results/endpoints_dictionary.npy', endpoints_dictionary)
    edge_gdf['signal_u_value'] = edge_gdf.signal_u.apply(lambda x: endpoints_dictionary[x])
    edge_gdf['signal_v_value'] = edge_gdf.signal_v.apply(lambda x: endpoints_dictionary[x])
    return edge_gdf

if __name__ == "__main__":
    os.environ['PATH'] += ":/home/shekhars/yang7492/.conda/envs/syntheticData/lib/python3.8/site-packages/sumo/bin"
    # Set SUMO_HOME
    os.environ['SUMO_HOME'] = '/home/shekhars/yang7492/.conda/envs/syntheticData/lib/python3.8/site-packages/sumo'
     
    # load SUMO map
    net_file = '../data/Minneapolis.net.xml'
    sumo_net = sumolib.net.readNet(net_file)

    # load openstreet map:
    osm_file_path = "../data/maps/minneapolis.graphml"
    osmnx_net = ox.io.load_graphml(osm_file_path)
    node_gdf, edge_gdf = ox.utils_graph.graph_to_gdfs(osmnx_net)
    edge_gdf = preprocess_edge(edge_gdf, node_gdf)
        
    # Define the input and output directories
    data_folder = "../data/synthetic/Murphy"
    output_folder = "../data/processed/Murphy"  # Make sure this exists or create it

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Pattern to match all CSV files in the data folder
    pattern = os.path.join(data_folder, "*.csv")

    # Iterate over all CSV files in the data folder
    for file_name in glob.glob(pattern):
        print(f"Processing {file_name}...")

        # Assuming read_data, edge_list_to_node_list, extracted_matched_edges, trip_to_edge_df are defined
        processed_df = read_data(file_name)
        processed_df['sumo_node_path'] = processed_df['sumo_path'].apply(lambda x: edge_list_to_node_list(x, sumo_net))
        processed_df = processed_df.apply(extracted_matched_edges, axis=1)

        df_edges = trip_to_edge_df(processed_df, edge_gdf)
        print(len(df_edges))

        # Generate output file name based on input file
        base_name = os.path.basename(file_name)
        output_file_name = os.path.join(output_folder, f"edge_level_{base_name}")

        # Save the processed DataFrame to a new CSV in the output folder
        df_edges.to_csv(output_file_name, index=False)

        print(f"Saved processed data to {output_file_name}")