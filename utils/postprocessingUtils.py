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


def cal_lanes(lanes_value):
    # Handle lists: assume you want to extract the first valid value
    if isinstance(lanes_value, list):
        # Filter out valid, numeric lane values, then take the first one
        valid_lanes = [lv for lv in lanes_value if pd.notna(lv) and str(lv).isdigit()]
        lanes_value = valid_lanes[0] if valid_lanes else 0

    # Handle floats and strings as before
    if pd.isna(lanes_value):
        return 0
    
    lanes_value_str = str(lanes_value)
    
    if lanes_value_str.isalpha():
        return 0
    
    if lanes_value_str.isalnum():
        # Extract digits and convert to integer, ensure non-negative
        numeric_part = ''.join(filter(str.isdigit, lanes_value_str))
        return int(numeric_part) if numeric_part and int(numeric_part) > 0 else 0
    else:
        # For non-alphanumeric strings, extract the first decimal value
        for i in lanes_value_str:
            if i.isdecimal():
                return int(i)
        return 0  # Default to 0 if no suitable value is found



def cal_bridge(bridge_value):
    # Handle lists: check each element
    if isinstance(bridge_value, list):
        # Initialize as not a bridge
        bridge_score = 0
        for item in bridge_value:
            if 'viaduct' in str(item).lower():
                bridge_score = max(bridge_score, 2)
            elif 'yes' in str(item).lower():
                bridge_score = max(bridge_score, 1)
        return bridge_score
    
    # Handle NaN values explicitly
    if pd.isna(bridge_value):
        return 0

    # Handle strings
    s = str(bridge_value).lower()  # Convert to string and lowercase to ensure consistency
    if 'viaduct' in s:
        return 2
    if 'yes' in s:
        return 1

    return 0  # Default return value if none of the conditions are met



def highway_cal(network_seg):
    '''Calcuate the road type of an edge ('unclassified' in default)

    Args:
        Attributes of an edge
    
    Returns:
        Road type of the edge 
    '''
    if 'highway' in network_seg and network_seg['highway']:
        if isinstance(network_seg['highway'], str):
            return network_seg['highway']
        elif isinstance(network_seg['highway'], list):
            return network_seg['highway'][0]
    else:
        return 'unclassified'
    
def length_cal(network_seg):
    '''Calcuate the length of an edge (-1 in default)

    Args:
        Attributes of an edge
    
    Returns:
        length (m) of the edge 
    '''
    if 'length' not in network_seg:
        return -1
    
    if pd.isna(network_seg['length']):
        return -1
    else:
        return network_seg['length']
    
def speedlimit_cal(network_seg):
    '''Calculate the speed limit of an edge (default to 30*1.60934 km/h if unspecified)

    Args:
        network_seg: Attributes of an edge
    
    Returns:
        Speed limit (km/h) of the edge 
    '''
    # Check if 'maxspeed' attribute exists and has a non-empty value
    if 'maxspeed' in network_seg and network_seg['maxspeed']:
        maxspeed = network_seg['maxspeed']
        # Handle different types of 'maxspeed' values
        if isinstance(maxspeed, list):
            # Assume the first element of the list is the relevant speed limit
            maxspeed = maxspeed[0]
        
        if isinstance(maxspeed, str):
            # Extract numeric part of the string
            numeric_speed = ''.join(filter(str.isdigit, maxspeed))
            if numeric_speed:
                # Convert to int and adjust to km/h
                return int(numeric_speed) * 1.60934
        
        elif isinstance(maxspeed, (int, float)):
            if not pd.isna(maxspeed):
                # Directly use the numeric value, assuming it's in mph and convert to km/h
                return maxspeed * 1.60934
    
    # Fallbacks based on highway type if 'maxspeed' is not specified
    elif highway_cal(network_seg) == "motorway":
        return 55 * 1.60934
    elif highway_cal(network_seg) == "motorway_link":
        return 50 * 1.60934
    
    # Default speed limit if none of the above conditions are met
    return 30 * 1.60934

def compute_direction_angle(network_seg):
    xs, ys = network_seg['geometry'].xy
    latitude_o,longitude_o = xs[0],ys[0]
    latitude_d,longitude_d  = xs[-1],ys[-1]
    direction = [latitude_d-latitude_o,longitude_d-longitude_o]
    direction_array = np.array(direction)
    cosangle = direction_array.dot(np.array([1,0]))/(np.linalg.norm(direction_array)) 
    if np.cross(direction_array,np.array([1,0])) < 0:
        direction_angle = math.acos(cosangle)*180/np.pi
    else :
        direction_angle = -math.acos(cosangle)*180/np.pi
    return direction_angle

def ori_cal(coor_a,coor_b,coor_c):
    '''Calcuate the orientation change from vector ab to bc (0 in default, right turn > 0, left turn < 0)

    10 degree is the threshold of a turn.

    Args:
        coor_a: coordinate of point a
        coor_b: coordinate of point b
        coor_c: coordinate of point c
    
    Returns:
        's': straight
        'l': left-hand turn
        'r': right-hand turn
    '''
    a = np.array(coor_a)
    b = np.array(coor_b)
    c = np.array(coor_c)
    v_ab = b-a
    v_bc = c-b
    cosangle = v_ab.dot(v_bc)/(np.linalg.norm(v_bc) * np.linalg.norm(v_ab)+1e-16) 
    return math.acos(cosangle)*180/np.pi if np.cross(v_ab,v_bc) < 0 else -math.acos(cosangle)*180/np.pi

def previous_orientation_cal(network_seg, edge_gdf, prev_edge_id):
    if prev_edge_id == -1 :
        orientation = 0
    elif 'geometry' in network_seg and network_seg['geometry'] and edge_gdf.loc[prev_edge_id,'geometry']:
        xs, ys = network_seg['geometry'].xy
        #print(id_before)
        xl, yl = edge_gdf.loc[prev_edge_id,'geometry'].xy
        coor_b = [xs[0],ys[0]]
        coor_c = [xs[1],ys[1]]
        coor_a = [xl[-2],yl[-2]]
        #print(beg,coor_a,coor_b,coor_c)
        orientation = ori_cal(coor_a,coor_b,coor_c)
    else:
        orientation = 0
    return orientation

def extract_osm_edge_feature(edge_gdf, edge_id, prev_edge_id):
    osm_edge_feature = edge_gdf.loc[edge_id]
    edge_features = {
        'tags': edge_id,
        'osmid': osm_edge_feature.osmid,
        'road_type': highway_cal(osm_edge_feature),
        'speed_limit': speedlimit_cal(osm_edge_feature),
        'length': length_cal(osm_edge_feature),
        'lanes': osm_edge_feature.lanes_normed,
        'bridge': osm_edge_feature.bridge_normed,
        'endpoint_u': osm_edge_feature.signal_u_value, 
        'endpoint_v': osm_edge_feature.signal_v_value,
        'direction_angle': compute_direction_angle(osm_edge_feature),
        'previous_orientation': previous_orientation_cal(osm_edge_feature, edge_gdf, prev_edge_id),
        'elevation_change': osm_edge_feature.elevation_change
    }
    return edge_features

def extract_simulated_feature(row, trip_id, position_of_edge, trip_start_index, trip_end_index, sumo_start_index, sumo_end_index):
    simulated_features = {
                    'trip_id': (0, trip_id),
                    'position': position_of_edge,
                    'mass': row['weight'][trip_start_index],
                    'energy_consumption_total_kwh': sum(row['total_fuel'][trip_start_index:trip_end_index])*40.3/3600/3.7854,
                    'simulated_energy_consumption_kwh': sum(row['fastsim_power'][sumo_start_index:sumo_end_index])/3600,
                    'time': trip_end_index - trip_start_index,
                    'sumo_time': sumo_end_index-sumo_start_index,
                    'speed': row['velocity_profile'][trip_start_index:trip_end_index],
                    'sumo_speed': row['sumo_velocity'][sumo_start_index:sumo_end_index],
                    'fastsim_speed': row['fastsim_velocity'][sumo_start_index:sumo_end_index],
                    'time_acc': row['trip_start_time'],
                    'time_stage': row['trip_start_time'].hour//4 + 1,
                    'week_day': row['trip_start_time'].weekday()+1,
                    'vehicle_type': row['vehicle_type']
                    }
    return simulated_features

def initialize_edge_df():

    # Define the column names based on the keys from both dictionaries
    columns = [
        'trip_id', 'position', 'mass', 'elevation_change', 'energy_consumption_total_kwh',
        'simulated_energy_consumption_kwh', 'time', 'sumo_time', 'speed', 'sumo_speed', 'fastsim_speed',
        'time_acc', 'time_stage', 'week_day', 'tags', 'osmid', 'road_type', 'speed_limit',
        'length', 'lanes', 'bridge', 'endpoint_u', 'endpoint_v', 'direction_angle',
        'previous_orientation', 'vehicle_type'
    ]

    # Initialize the DataFrame with these columns
    df_edges = pd.DataFrame(columns=columns)

    return df_edges

def index_range_in_sumo(edge_id, row, sumo_start_index):
    new_sumo_start_index = sumo_start_index - 1
    new_sumo_end_index = sumo_start_index - 1
    if sumo_start_index < len(row["sumo_node_path"]):
        for i in range(sumo_start_index, len(row["sumo_node_path"])):
            if str(edge_id[0]) in row["sumo_node_path"][i][0]:
                new_sumo_start_index = i
                break
        if new_sumo_start_index < sumo_start_index:
            return sumo_start_index, sumo_start_index + 1, False
        for i in range(new_sumo_start_index, len(row["sumo_node_path"])):
            if str(edge_id[1]) in row["sumo_node_path"][i][1] and (i >= len(row["sumo_node_path"]) - 1 or row["sumo_node_path"][i][1] != row["sumo_node_path"][i+1][1]):
                new_sumo_end_index = i
                return new_sumo_start_index, new_sumo_end_index + 1, True
    return sumo_start_index, sumo_start_index + 1, False

def trip_to_edge_df(processed_df, edge_gdf):
    df_edges = initialize_edge_df()
    for i in range(len(processed_df)):
        row = processed_df.iloc[i]
        # these indics are exclusive: [start_index, end_index]
        trip_start_index = 0
        trip_end_index = 0

        # if the vehicle is not moving at the start of the trip, skip these data loggings.
        if row['velocity_profile'][trip_start_index] <= 1e-10:
            trip_start_index += 1
            trip_end_index += 1

        sumo_start_index = 0
        sumo_end_index = 0

        position_of_edge = 0
        prev_edge_id = -1

        while trip_start_index < len(row["road_id"]):

            edge_id = row["road_id"][trip_start_index]
            # validate edge_id by checking if both the origin and the destinaiton occur in the sumo_node_path

            # exclusive index
            while trip_end_index < len(row["road_id"]) and row["road_id"][trip_end_index] == edge_id:
                trip_end_index += 1

            # exclusive index
            # for a node pair in road_id, if that both the origin node and the dest node exist in sumo_path, then it is a valid edge for training 
            sumo_start_index, sumo_end_index, found_flag = index_range_in_sumo(edge_id, row, sumo_start_index)

            # if the edge_id is in the sumo simulation
            if found_flag:

                # extract simulated_features
                edge_features = extract_osm_edge_feature(edge_gdf, edge_id, prev_edge_id)
                simulated_features = extract_simulated_feature(row, i, position_of_edge, trip_start_index, trip_end_index, sumo_start_index, sumo_end_index)
                
                # Combine edge_features and additional_features
                edge_row = {**edge_features, **simulated_features}

                # Add the combined features as a new row to the DataFrame
                df_edges.loc[len(df_edges)] = edge_row

                # update the sumo_start_index search range
                sumo_start_index = sumo_end_index
                position_of_edge += 1

            prev_edge_id = edge_id

            # update the trip_start_index anyway
            trip_start_index = trip_end_index
    return df_edges