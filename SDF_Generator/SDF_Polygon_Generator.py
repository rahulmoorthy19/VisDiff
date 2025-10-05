import numpy as np
import pickle
import os
import cv2
import matplotlib.pyplot as plt
from collections import deque
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import h5py
import geopandas as gpd
import time
import dask_geopandas as dgpd
import copy
import joblib
import random
random.seed(123)
np.random.seed(123)
def bfs_signed_distance(all_coordinates, polygon):
    poly = Polygon(polygon)
    points_gdf = dgpd.from_geopandas(gpd.GeoSeries([Point(pt) for pt in all_coordinates]), npartitions=12)
    distances = points_gdf.distance(poly.boundary).compute()
    inside_mask = points_gdf.within(poly).compute()
    signed_distances = np.where(inside_mask, -1 * np.clip(distances, -1, 1), np.clip(distances, -1, 1))
    return signed_distances

def sdf_converter(all_coordinates, polygon):
    sdf = bfs_signed_distance(all_coordinates, polygon)
    return sdf


count = 0
## Change the path to the required dataset path
dataset_directory = ""
## Keep the same as dataset directory
save_dataset_directory = ""
data_list = [path for path in sorted(os.listdir(dataset_directory)) if not os.path.isdir(dataset_directory + path)]
random.shuffle(data_list)
## Change the resolution to the required resolution
grid_resolution = 0.01
x_max = 2.2
y_max = 2.2
print("started")

for each_file in data_list:
    if os.path.isfile(dataset_directory + each_file):
        start = time.time()
        polygon_data_batch = joblib.load(dataset_directory + each_file)
        for polygon_dict_key in polygon_data_batch.keys():
            polygon_dict = polygon_data_batch[polygon_dict_key]
            polygon = polygon_dict["level_2_polygon"]
            ## Smaller Resolution
            x = np.linspace(0, x_max, int(x_max/ grid_resolution))
            y = np.linspace(0, y_max, int(y_max/ grid_resolution))
            X_mesh, Y_mesh = np.meshgrid(x, y)
            X_flat = X_mesh.flatten()
            Y_flat = Y_mesh.flatten()
            all_coordinates = np.column_stack((X_flat, Y_flat))
            polygon_sdf_image = sdf_converter(all_coordinates, polygon)

            indices = np.stack(np.indices(X_mesh.shape), axis=-1).reshape(-1, 2)
            sdf_mask = np.zeros(X_mesh.shape)
            sdf_mask[indices[:, 0], indices[:, 1]] = polygon_sdf_image
            polygon_dict["small_continuous_sdf_image"] = sdf_mask

        print(f"Processed {each_file} (Time: {time.time() - start:.2f} sec)")
        batch_file = save_dataset_directory + each_file
        joblib.dump(polygon_data_batch, batch_file, compress=3)
        print(f"Saved {batch_file} with {len(polygon_data_batch)} files")
