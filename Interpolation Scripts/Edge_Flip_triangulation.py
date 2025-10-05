from sect.triangulation import Triangulation
from ground.base import get_context
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import triangle
from collections import defaultdict
import copy 


def create_triangulation(polygon):
    edges = list()
    for each in range(polygon.shape[0]):
        edges.append([each, (each + 1)%polygon.shape[0]])
    A = dict(vertices=polygon, segments=edges)  
    tri = triangle.triangulate(A, 'p')
    triangulation = tri["triangles"]
    return triangulation

def triangulation_to_edges(triangles):
    edges = set()
    for tri in triangles:
        for i in range(3):
            a, b = sorted((tri[i], tri[(i + 1) % 3]))
            edges.add((a, b))
    return edges


def triangulation_shared_edges(triangles):
    non_shared_edges = set()
    shared_edges = list()
    for tri in triangles:
        for i in range(3):
            a, b = sorted((tri[i], tri[(i + 1) % 3]))
            if (a,b) in non_shared_edges:
                shared_edges.append((a,b))
            non_shared_edges.add((a, b))
    non_shared_edges = list(non_shared_edges)
    for each_edge in shared_edges:
        non_shared_edges.remove(each_edge)
    return non_shared_edges, shared_edges 

def get_adjacent_triangles(triangles):
    edge_map = defaultdict(list)
    for idx, tri in enumerate(triangles):
        for i in range(3):
            a, b = sorted((tri[i], tri[(i + 1) % 3]))
            edge_map[(a, b)].append(idx)
    return edge_map
    
def flip_edge(triangles, edge):
    edge_map = get_adjacent_triangles(triangles)
    tris = edge_map[edge]
    if len(tris) != 2:
        return triangles  # can't flip boundary edge

    t1, t2 = triangles[tris[0]], triangles[tris[1]]
    # Get the opposite vertices
    opp1 = list(set(t1) - set(edge))[0]
    opp2 = list(set(t2) - set(edge))[0]

    # Perform the flip
    new_tri1 = [opp1, opp2, edge[0]]
    new_tri2 = [opp1, opp2, edge[1]]

    new_tris = triangles.copy()
    new_tris[tris[0]] = new_tri1
    new_tris[tris[1]] = new_tri2
    return new_tris

def triangulations_equal(T1, T2):
    return set(map(lambda tri: tuple(sorted(tri)), T1)) == set(map(lambda tri: tuple(sorted(tri)), T2))

def convert_to_adjacency(triangles, no_of_vertices = 25):
    non_shared_edges, shared_edges = triangulation_shared_edges(triangles)
    adj_mat = np.zeros((no_of_vertices, no_of_vertices))
    for each_edge in shared_edges:
        cur_edge_1, cur_edge_2 = each_edge[0], each_edge[1]
        adj_mat[cur_edge_1, cur_edge_2] = 1
        adj_mat[cur_edge_2, cur_edge_1] = 1

    for each_edge in non_shared_edges:
        cur_edge_1, cur_edge_2 = each_edge[0], each_edge[1]
        adj_mat[cur_edge_1, cur_edge_2] = 1
        adj_mat[cur_edge_2, cur_edge_1] = 1
    
    for i in range(no_of_vertices):
        adj_mat[i,i] = 1
    return adj_mat

def array_in_list(target, array_list):
    return any(np.array_equal(target, arr) for arr in array_list)

def valid_triangulation_steps(polygon_start, no_of_flips = 0):
    T_start = create_triangulation(polygon_start)
    T_current = T_start.copy()
    intermediate_adj_mat = list()
    trig_g = convert_to_adjacency(T_current)
    intermediate_adj_mat.append(trig_g)
    for each_flip in range(no_of_flips):
        while True:
            non_shared_edges, shared_edges = triangulation_shared_edges(T_current)
            flipping_edge = np.random.choice([i for i in range(len(shared_edges))], size = 1)[0]
            edge = shared_edges[flipping_edge]
            T_new = flip_edge(T_current, edge)
            if not triangulations_equal(T_new, T_current):
                T_current = copy.deepcopy(T_new)
                trig_g = convert_to_adjacency(T_current)
                if not array_in_list(trig_g, intermediate_adj_mat):
                    intermediate_adj_mat.append(trig_g)
                    break
    return intermediate_adj_mat

def transform_triangulation(polygon_start, polygon_end):
    T_start = create_triangulation(polygon_start)
    T_goal = create_triangulation(polygon_end)
    intermediate_adj_mat = list()
    current_triangulation = T_start
    visited = set()
    visited.add(tuple(map(tuple, current_triangulation)))  # Add the current triangulation as a tuple of tuples
    intermediate_adj_mat.append(convert_to_adjacency(T_start))
    flips = []
    # Perform edge flips until we reach the goal triangulation
    while not triangulations_equal(current_triangulation, T_goal):
        for edge in triangulation_to_edges(current_triangulation):
            if edge not in triangulation_to_edges(T_goal):  # Edge not in target triangulation
                new_triangulation = flip_edge(current_triangulation, edge)
                
                # Check if the new triangulation has been visited
                if tuple(map(tuple, new_triangulation)) not in visited:
                    current_triangulation = new_triangulation
                    visited.add(tuple(map(tuple, new_triangulation)))  # Add to visited set
                    flips.append(edge)
                    intermediate_adj_mat.append(convert_to_adjacency(current_triangulation))
                    break

    return current_triangulation, intermediate_adj_mat, T_goal, T_start, convert_to_adjacency(T_start), convert_to_adjacency(T_goal)

# with open("/home/isleri/mahes092/Polygon_Dataset_Generator/Minimum_Feature_25_Dataset_Test_New/25_0_coords.pkl", "rb") as f:
#     polygon_dict = pickle.load(f)

# with open("/home/isleri/mahes092/Polygon_Dataset_Generator/Minimum_Feature_25_Dataset_Test_New/25_10_coords.pkl", "rb") as f:
#     polygon_dict_1 = pickle.load(f)

# valid_triangulation, flips, T_goal = transform_triangulation(polygon_dict["level_2_polygon"], polygon_dict_1["level_2_polygon"])

# print(valid_triangulation, T_goal)