import os
import glob
import ast
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import datetime
import time
from shapely.geometry import Polygon
import gc
from os import walk
import geopandas as gpd
import math
import random
from random import shuffle
import csv
import pickle

def feature_extraction(path_to_data_folder, path_to_results_folder, edge_level_file_pattern, output_file_folder, synthetic_data_flg=True):


    # Get a list of all files to process
#     file_pattern = os.path.join(path_to_data_folder, edge_level_file_pattern)
    file_pattern = edge_level_file_pattern
    matched_files = glob.glob(file_pattern)

    # Specify the destination folder for processed files
#     output_folder = os.path.join(path_to_data_folder, output_file_folder)
    output_folder = output_file_folder
    os.makedirs(output_folder, exist_ok=True)  # Create the destination folder if it doesn't exist

    # file paths
    dual_graph_nodes_file_name = os.path.join(path_to_results_folder, "dualGraphNodes.pkl")
    outputpath_prefix = os.path.join(output_folder, "syntheticData") if synthetic_data_flg else os.path.join(output_folder, "realData")
    roadtype_output_root = os.path.join(path_to_results_folder, "road_type_dictionary.csv")
    endpoint_output_root = os.path.join(path_to_results_folder, "endpoints_dictionary.csv")
    stats_output_root = os.path.join(path_to_results_folder, "mean_std.csv")

    if not os.path.exists(outputpath_prefix):
        os.mkdir(outputpath_prefix)

    # all raw files
    file_cnt = 0
    matched_files.sort()
    for filename in matched_files:
        if file_cnt == 0:
            df_edge = pd.read_csv(filename)
            df_edge['trip_id'] = df_edge['trip_id'].apply(ast.literal_eval)
        else:
            df = pd.read_csv(filename)
            df['trip_id'] = df['trip_id'].apply(ast.literal_eval)
            df.trip_id = df.trip_id.apply(lambda x: (file_cnt, x[1]))
            df_edge = pd.concat([df_edge,df], ignore_index=True)
        file_cnt += 1



    columns_to_process = ['trip_id', 'position', 'mass', 'elevation_change',
           'energy_consumption_total_kwh',   'time', 'speed', 'time_acc',
           'time_stage', 'week_day', 'tags', 'osmid', 'road_type', 'speed_limit',
           'length', 'lanes', 'bridge', 'endpoint_u', 'endpoint_v',
           'direction_angle', 'previous_orientation', 'vehicle_type']
    if synthetic_data_flg:
        columns_to_process[4] = 'simulated_energy_consumption_kwh'
        columns_to_process[5] = 'sumo_time'
        columns_to_process[6] = 'fastsim_speed'
        df_edge = df_edge[columns_to_process]
        newNames = {'simulated_energy_consumption_kwh': 'energy_consumption_total_kwh', 'sumo_time': 'time', 'fastsim_speed': 'speed'}
        df_edge = df_edge.rename(columns=newNames)
    else:
        df_edge = df_edge[columns_to_process]


    df_edge['osmNodeIdUV'] = df_edge.tags.apply(lambda x: tuple(list(map(int, x[1:-1].split(", ")))[:-1]))

    with open(dual_graph_nodes_file_name, "rb") as open_file:
        dualGraphNode = pickle.load(open_file)

    df_edge['osmNode'] = df_edge.osmNodeIdUV.apply(lambda x: dualGraphNode.index((x[0], x[1], 0)))
    df_edge = df_edge.fillna(axis=0,method='ffill')
    df_edge['segment_count'] = df_edge.groupby('trip_id')['osmNodeIdUV'].transform('count')
    df_edge['network_id'] = df_edge['osmid']

    #remove extreme short trips
    df_edge = df_edge.drop(df_edge[df_edge['segment_count']<3].index)

    df_edge = df_edge.reset_index(drop = True)

    # remove edges with extreme large elevation change
    per_ele_001 = df_edge.elevation_change.quantile(0.01)
    per_ele_99 = df_edge.elevation_change.quantile(0.99)
    df_edge = df_edge.drop(df_edge[df_edge['elevation_change'] >per_ele_99].index).\
        drop(df_edge[df_edge['elevation_change'] < per_ele_001].index).reset_index(drop = True)

    counterFunc = df_edge.apply(lambda x: True if abs(x['previous_orientation']) > 179 else False, axis=1)
    df_edge.drop(counterFunc[counterFunc == True].index,inplace=True)
    df_edge.reset_index(drop = True, inplace = True)


    cnt = 0
    for i in range(len(df_edge)):
        if i > 0 and df_edge.loc[i,'trip_id'] != df_edge.loc[i-1,'trip_id']:
            cnt += 1
        df_edge.loc[i,'trip']  = cnt


    random.seed(1234)
    trip_num = len(df_edge['trip_id'].unique())
    k_folder_list = list(range(trip_num))
    shuffle(k_folder_list)

    test_list  = k_folder_list[int(0.8*len(k_folder_list)):]
    k_folder_list = k_folder_list[:int(0.8*len(k_folder_list))]    


    df_edge = df_edge[['network_id', 'position',
           'road_type', 'speed_limit', 'mass', 'elevation_change',
           'previous_orientation', 'length', 'energy_consumption_total_kwh', 
           'time',  'direction_angle', 'time_stage', 'week_day',
            'lanes', 'bridge', 'endpoint_u', 'endpoint_v', 'segment_count', 'trip','osmNodeIdUV','osmNode','vehicle_type']]

    df_edge['segment_count'] = df_edge.groupby('trip')['network_id'].transform('count')

    trip_before = -1
    position = 1
    for i in range(len(df_edge)):
        if df_edge.loc[i,'trip'] != trip_before:
            position = 1
            trip_before = df_edge.loc[i,'trip']
        else:
            position += 1
        df_edge.loc[i,'position']  = position

    d = df_edge.groupby('road_type')['speed_limit'].mean()

    d.sort_values()

    dictionary = {}
    road_tp = 0
    for i in d.sort_values().index:
        dictionary[i] = road_tp
        road_tp += 1


    csvFile = open(roadtype_output_root, "w")
    writer = csv.writer(csvFile)
    writer.writerow(["road type", "value"])
    for i in dictionary:
        writer.writerow([i, dictionary[i]])
    csvFile.close()
    np.save(os.path.join(path_to_results_folder,'road_type_dictionary.npy'), dictionary)
    endpoints_dictionary = np.load(os.path.join(path_to_results_folder, 'endpoints_dictionary.npy'), allow_pickle=True).item()


    csvFile = open(endpoint_output_root, "w")
    writer = csv.writer(csvFile)
    writer.writerow(["endpoint", "value"])
    for i in endpoints_dictionary:
        writer.writerow([i, endpoints_dictionary[i]])
    csvFile.close()

    df_edge['road_type']=df_edge['road_type'].apply(lambda x:dictionary[x])

    # for lookuptable method
    #     output = outputFolderPrefix + str(datasetenumerate)
    #     outputpath = os.path.join("lookupdata", output)
    #     print(outputpath)
    #     if not os.path.exists(outputpath):
    #         os.mkdir(outputpath)
    #     # df_train = df_test[df_test['trip'].apply(lambda x: x in train_list_3 or x in val_list_3)]
    #     df_train = df_test[df_test['trip'].apply(lambda x: x in train_list_3)]
    #     df_val = df_test[df_test['trip'].apply(lambda x: x in test_list)]
    #     df_train.to_csv(os.path.join(outputpath,"train_data.csv"))
    #     df_val.to_csv(os.path.join(outputpath,"val_data.csv"))

    #     print('lookuptable finished')
    new_columns = [
     'speed_limit',
     'mass',
     'elevation_change',
     'previous_orientation',
     'length',
     'direction_angle',
     'network_id',
     'position',
     'road_type',
     'time_stage',
     'week_day',
     'lanes',
     'bridge',
     'endpoint_u',
     'endpoint_v',
     'energy_consumption_total_kwh',
     'time',
     'segment_count',
     'trip',
     'osmNodeIdUV',
     'osmNode',
     'vehicle_type'
        ''
    ]

    df02 = df_edge.reindex(columns=new_columns)

    csvFile = open(stats_output_root, "w")
    writer = csv.writer(csvFile)
    writer.writerow(["attribute", "mean","std"])
    for i,val in enumerate(df02.columns):
        if i < 6:
            x_mean = df02[val].mean()
            x_std = df02[val].std()
            writer.writerow([val,x_mean,x_std])
            df02[val] = df02[val].apply(lambda x: (x - x_mean) / x_std)
        elif val == 'energy_consumption_total_kwh' or val == 'time':
            x_mean = df02[val].mean()
            x_std = df02[val].std()
            writer.writerow([val, x_mean, x_std])
    csvFile.close()

    for datasetenumerate in range(1,11):

        outputpath = os.path.join(outputpath_prefix, str(datasetenumerate))
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)

        random.seed(datasetenumerate)
        shuffle(k_folder_list)
        #60-20-20
        train_list  = k_folder_list[: int(0.75*len(k_folder_list))]
        val_list  = k_folder_list[int(0.75*len(k_folder_list)):]

#         print(len(train_list), len(val_list), len(test_list))

        df_train = df02[df02['trip'].apply(lambda x: x in train_list)]
        df_val  = df02[df02['trip'].apply(lambda x: x in val_list)]
        df_t = df02[df02['trip'].apply(lambda x: x in test_list)]

        file_name_list = ["train_data.csv", "val_data.csv", "test_data.csv"]
        file_cnt = 0
        for df in [df_train, df_val, df_t]:
            df = df.fillna(axis=0,method='ffill')
            df.reset_index(drop = True, inplace = True)
            df['data'] = df.apply(lambda x: [x['speed_limit'],x['mass'],x['elevation_change'],x['previous_orientation'],x['length'],x['direction_angle']], axis = 1)
            df['label'] = df.apply(lambda x: [x["energy_consumption_total_kwh"],x["time"]], axis = 1)
            trip_before = -1
            position = 1
            for i in range(len(df)):
                if df.loc[i,'trip'] != trip_before:
                    position = 1
                    trip_before = df.loc[i,'trip']
                else:
                    position += 1
                df.loc[i,'position_new'] = position
            df['trip'] = df['trip'].apply(lambda x: int(x))
            df = df[['data','label','vehicle_type','segment_count',"position_new","road_type","time_stage", "week_day", "lanes", "bridge", "endpoint_u", "endpoint_v","trip",'osmNode']]
            file = file_name_list[file_cnt]
            file_cnt += 1
            df.to_csv(os.path.join(outputpath,file),header=False, index = False)
            
if __name__ == "__main__":
    path_to_data_folder = "../data"
    path_to_results_folder = "../results"
    # extract features for neural network training
    edge_level_file_pattern = os.path.join(edge_level_output_folder, "*_processed.csv")
    output_file_folder =  os.path.join(data_file_folder, "features/Murphy")
    # process synthetic data or real-world data 
    synthetic_data_flg = True # or False for processing real-world data
    feature_extraction(path_to_data_folder, path_to_results_folder, edge_level_file_pattern, output_file_folder, synthetic_data_flg)