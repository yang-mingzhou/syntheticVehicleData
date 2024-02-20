from mappymatch import package_root
import pandas as pd
import ast
import os
import re

def preprocess_vehicle_data(vehicle_file, gps_file):
    # Specify data types if known, otherwise set low_memory=False
    vehicle_data = pd.read_csv(vehicle_file, header=0, usecols=[0, 2, 3, 8, 12, 15, 16, 18, 19, 20, 21, 22, 23, 24], parse_dates=[[8, 9, 10, 11, 12, 13]], low_memory=False)
    vehicle_data.drop(vehicle_data.index[0], inplace=True)
    cols = ['Time_abs', 'Time', 'AmbAirTemp', 'BarPressure', 'EngineFuel', 'inclination', 'VehicleSpeed', 'VehicleWeight', 'FileIndex']
    vehicle_data.columns = cols

    gps_data = pd.read_csv(gps_file, low_memory=False)
    gps_data.drop(gps_data.index[0], inplace=True)
    gps_data = gps_data.astype(float)

    merged_data = pd.concat([vehicle_data, gps_data], axis=1)
    
    # Downsample to 1 Hz by taking every 10th row
    merged_data = merged_data.iloc[::10, :]
    
    min_latitude, max_latitude = -90, 90
    min_longitude, max_longitude = -180, 180
    valid_gps = merged_data[(merged_data['gps_Latitude'].between(min_latitude, max_latitude)) & 
                            (merged_data['gps_Longitude'].between(min_longitude, max_longitude))]

    # Convert FileIndex to boolean (True if 1, False otherwise)
    
    valid_gps.loc[:, 'Time_abs'] = valid_gps['Time_abs'].apply(lambda x: x if '.' in x else x + '.0')
    valid_gps.loc[:, 'Time_abs'] = pd.to_datetime(valid_gps['Time_abs'], format='%Y %m %d %H %M %S.%f', errors='coerce')
    valid_gps.loc[:, ['Time', 'AmbAirTemp', 'BarPressure', 'EngineFuel', 'inclination', 'VehicleSpeed', 'VehicleWeight','FileIndex']] = valid_gps[['Time', 'AmbAirTemp', 'BarPressure', 'EngineFuel', 'inclination', 'VehicleSpeed', 'VehicleWeight','FileIndex']].astype("float")
    valid_gps['FileIndex'] = valid_gps['FileIndex'] == 1
    #     valid_gps.loc[:, 'VehicleWeight'] = valid_gps['VehicleWeight'].apply(lambda x: 12500 if x < 12500 else (36500 if x > 36500 else x))
    

    return valid_gps

def convert_to_trip_based(df, time_gap_threshold=2.0):
    df['time_diff'] = df['Time_abs'].diff().dt.total_seconds()

    # Identify start of a new trip
    df['trip_start'] = df['FileIndex'] | (df['time_diff'] > time_gap_threshold)
    df['trip_id'] = df['trip_start'].cumsum()

    # Aggregate data for each trip
    trip_data = df.groupby('trip_id').agg({
        'Time_abs': ['first', 'last', lambda x: (x.max() - x.min()).total_seconds()],
        'gps_Latitude': lambda x: list(x),
        'gps_Longitude': lambda x: list(x),
        'gps_Altitude': list,
        'VehicleSpeed': list,
        'VehicleWeight': list,
        'EngineFuelRate': list,  # Converting fuel rate from l/h to total liters
        'AmbAirTemp': list
    })

    # Renaming columns
    trip_data.columns = ['trip_start_time', 'trip_end_time', 'travel_time', 'latitudes', 'longitudes', 'altitude_profile', 'velocity_profile', 'weight', 'total_fuel', 'ambTemperature']

    # Combine latitude and longitude into trajectory
    trip_data['trajectory'] = trip_data.apply(lambda row: list(zip(row['latitudes'], row['longitudes'])), axis=1)

    # Drop the separate latitude and longitude columns
    trip_data.drop(['latitudes', 'longitudes'], axis=1, inplace=True)

    return trip_data

def process_and_save_trip_data(vehicle_file, gps_file, output_csv_file):
    # Preprocess and downsample data
    processed_data = preprocess_vehicle_data(vehicle_file, gps_file)

    # Convert to trip-based data
    trip_based_data = convert_to_trip_based(processed_data)

    # Save trip_data to a CSV file
    trip_based_data.to_csv(output_csv_file, index=False)

    print(f"Trip data saved to {output_csv_file}")
    
def read_trip_data(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert string representations of lists back to lists
    df['trajectory'] = df['trajectory'].apply(ast.literal_eval)
    df['velocity_profile'] = df['velocity_profile'].apply(ast.literal_eval)
    df['altitude_profile'] = df['altitude_profile'].apply(ast.literal_eval)

    return df


def process_all_files_in_directory(base_dir):
    # Construct the path for the output folder
    output_base_dir = os.path.join(os.path.dirname(base_dir), 'tripsData (Murphy)')

    # Create the output folder if it does not exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Walk through all files and subdirectories in the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if the file is a vehicle data file
            if re.match(r'TL\d+-\d+_\d+W\d+\.csv', file) and not 'gps' in file:
                vehicle_file = os.path.join(root, file)
                gps_file = vehicle_file.replace('.csv', '_gps.csv')

                # Check if the corresponding GPS file exists
                if os.path.exists(gps_file):
                    # Construct the output file path in the output folder
                    output_file_name = os.path.basename(vehicle_file).replace('.csv', '_trip_data.csv')
                    output_csv_file = os.path.join(output_base_dir, output_file_name)

                    process_and_save_trip_data(vehicle_file, gps_file, output_csv_file)
                    print(f"Processed {vehicle_file} and {gps_file}, saved to {output_csv_file}")
                else:
                    print(f"GPS file not found for {vehicle_file}")

if __name__ == "__main__":
    # Example usage
    base_directory = "./Baseline Data (Murphy)"
    process_all_files_in_directory(base_directory)


    # # Example usage for one file
    # vehicle_file = "data/exampleOBD/TL5-231_2021W6.csv"
    # gps_file = "data/exampleOBD/TL5-231_2021W6_gps.csv"

    # # Preprocess and downsample data
    # processed_data = preprocess_vehicle_data(vehicle_file, gps_file)

    # # Convert to trip-based data
    # trip_based_data = convert_to_trip_based(processed_data)
    # print(trip_based_data.head())

    # process_and_save_trip_data('data/exampleOBD/TL5-231_2021W6.csv', 'data/exampleOBD/TL5-231_2021W6_gps.csv', 'results/output_trip_data.csv')


    # # Example usage
    # trip_data_read = read_trip_data('results/output_trip_data.csv')
    # print(trip_data_read.head())

