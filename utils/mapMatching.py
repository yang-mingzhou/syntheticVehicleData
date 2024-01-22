import ast
import pandas as pd
import os
import json
import numpy as np

from mappymatch import package_root
from mappymatch.constructs.trace import Trace
from mappymatch.utils.plot import plot_trace
from mappymatch.constructs.geofence import Geofence
from mappymatch.utils.plot import plot_geofence
from mappymatch.maps.nx.nx_map import NxMap, NetworkType
from mappymatch.utils.plot import plot_map
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from mappymatch.matchers.valhalla import ValhallaMatcher
from mappymatch.matchers.osrm import OsrmMatcher
from mappymatch.utils.plot import plot_matches
from mappymatch.utils.plot import plot_path

def read_trip_data(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert string representations of lists back to lists
    df['trajectory'] = df['trajectory'].apply(ast.literal_eval)
    df['velocity_profile'] = df['velocity_profile'].apply(ast.literal_eval)
    df['altitude_profile'] = df['altitude_profile'].apply(ast.literal_eval)

    return df


def bounding_box_to_geojson(bounding_box, output_file):
    # Define the coordinates of the bounding box (Polygon)
    # The coordinates list must start and end at the same point, forming a closed loop
    coordinates = [
        [
            [bounding_box['min_longitude'], bounding_box['min_latitude']],  # Lower-left corner
            [bounding_box['min_longitude'], bounding_box['max_latitude']],  # Upper-left corner
            [bounding_box['max_longitude'], bounding_box['max_latitude']],  # Upper-right corner
            [bounding_box['max_longitude'], bounding_box['min_latitude']],  # Lower-right corner
            [bounding_box['min_longitude'], bounding_box['min_latitude']]   # Closing the loop at lower-left corner
        ]
    ]
    # Define the GeoJSON structure
    geojson_object = {
        "type": "Feature",
        "properties": {},  # Properties can be added if needed
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        }
    }    
    # Write the GeoJSON object to a file
    with open(output_file, 'w') as f:
        json.dump(geojson_object, f, indent=4)
    
    print(f"GeoJSON file saved to {output_file}")
       
        
def is_within_bounding_box(trajectory, bounding_box):
    # Check if all points in the trajectory are within the bounding box
    return all(
        bounding_box['min_latitude'] <= lat <= bounding_box['max_latitude'] and
        bounding_box['min_longitude'] <= lon <= bounding_box['max_longitude']
        for lat, lon in trajectory
    )


def geofence_from_bbox(bounding_box):
    output_geojson_file = 'results/bounding_box_mn.geojson'
    bounding_box_to_geojson(bounding_box, output_geojson_file)
    geofence = Geofence.from_geojson(output_geojson_file)
    return geofence


def traces_from_trips(filtered_trips):
    '''generate a list of traces for all filtered trips'''
    # Initialize an empty list to store Trace objects
    traces = []
    # Iterate through each trajectory in the filtered_trips DataFrame
    for trajectory in filtered_trips['trajectory']:
        # Convert the list of tuples (latitude, longitude) into a DataFrame
        trajectory_df = pd.DataFrame(trajectory, columns=['latitude', 'longitude'])  
        # Create a Trace object from the trajectory DataFrame
        trace = Trace.from_dataframe(trajectory_df, lat_column="latitude", lon_column="longitude")
        # Append the Trace object to the list of traces
        traces.append(trace)
    return traces

def postprocessing(filtered_trips, batch_match_result):
    # Initialize an empty list to store the start-end tuples for each trip
    start_end_tuples_list = []
    cooridinate_id_list = []
    road_id_list = []

    # Initialize an empty list to store the indices of trips with valid map-matching results
    valid_trip_indices = []

    for i, match_result in enumerate(batch_match_result):
        if match_result is not None and not match_result.matches_to_dataframe()['road_id'].isna().all():
            # Extract the start-end tuples from the match_result
            start_end_tuples = [(road.road_id.start, road.road_id.end, road.road_id.key) for road in match_result.path]
            match_df = match_result.matches_to_dataframe()

            # Drop rows where 'road_id' is NaN
            match_df = match_df.dropna(subset=['road_id'])
            cooridinate_id_list.append(match_df['coordinate_id'].to_list())
            
            # Transform 'road_id' column to a tuple of (start, end, key)
            match_df['road_id'] = match_df['road_id'].apply(lambda x: (x.start, x.end, x.key) if x is not None else None)
            road_id_list.append(match_df['road_id'].to_list())

            # Add the start-end tuples to the list
            start_end_tuples_list.append(start_end_tuples)

            # Record the index of the trip with a valid map-matching result
            valid_trip_indices.append(i)
        else:
            # If all road_id are NaN, we will not include this trip in the final DataFrame
            continue

    # Filter the filtered_trips DataFrame to only include trips with valid map-matching results
    filtered_trips = filtered_trips.iloc[valid_trip_indices]

    # Add the start-end tuples list as a new column to the filtered_trips DataFrame
    filtered_trips['matched_path'] = start_end_tuples_list
    filtered_trips['coordinate_id'] = cooridinate_id_list
    filtered_trips['road_id'] = road_id_list
    return filtered_trips


def process_and_save_trip_data(input_csv_file, output_csv_file, bounding_box, matcher):
    trip_data_read = read_trip_data(input_csv_file)

    # Apply the is_within_bounding_box function to each trajectory
    inside_mask = trip_data_read['trajectory'].apply(is_within_bounding_box, bounding_box=bounding_box)

    # Filter the DataFrame to only include trips inside the bounding box
    filtered_trips = trip_data_read[inside_mask]

    # Match all the trips
    traces = traces_from_trips(filtered_trips)
    batch_match_result = [matcher.match_trace(traces[i]) for i in range(len(traces))]

    # Postprocess the matched trips
    filtered_trips = postprocessing(filtered_trips, batch_match_result)

    # Save the processed data to a CSV file
    filtered_trips.to_csv(output_csv_file, index=False)

    
def process_all_files(input_directory, output_directory, bounding_box, matcher):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get a list of all CSV files in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith('_trip_data.csv'):
            input_csv_file = os.path.join(input_directory, file_name)
            output_csv_file = os.path.join(output_directory, file_name.replace('_trip_data.csv', '_matched.csv'))
            # Process each file
            process_and_save_trip_data(input_csv_file, output_csv_file, bounding_box, matcher)
            

if __name__ == "__main__":
    # define the bounding box
    bounding_box = {
        'min_latitude': 44.403672,
        'max_latitude': 45.450154,
        'min_longitude': -94.073366,
        'max_longitude': -92.696696
    }
    
    # download map
    geofence = geofence_from_bbox(bounding_box)
    nx_map = NxMap.from_geofence(geofence, network_type=NetworkType.DRIVE)
    matcher = LCSSMatcher(nx_map, distance_epsilon = 100, distance_threshold = 500)

    input_directory = "../data/trips/Murphy"
    output_directory = "../data/matchedTrips/Murphy"
    process_all_files(input_directory, output_directory, bounding_box, matcher)
