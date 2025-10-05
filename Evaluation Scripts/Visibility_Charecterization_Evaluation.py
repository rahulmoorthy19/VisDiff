import torch
import copy
import numpy as np
import pickle
from itertools import tee
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from multiprocessing import Pool
from skimage.morphology import skeletonize, thin
import os
import math
import cv2
import visvalingamwyatt as vw
# from Polygon_Fitting_batch_latest import Visibility_Optimization_Model
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from collections import deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)  # This sets the seed for all GPUs

class Schedule:
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas
    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]
    def __len__(self) -> int:
        return len(self.sigmas)
    def sample_batch(self, x0:torch.FloatTensor) -> torch.FloatTensor:
        return self[torch.randint(len(self), (x0.shape[0],))].to(x0)
    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
        indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                       .round().astype(np.int64) - 1)
        return self[indices + [0]]
    
class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float=10):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))

def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

@torch.no_grad()
def samples(model, condition, sigmas, gam, mu, xt, batchsize):
    start = None
    if xt is None:
        xt = (torch.randn((batchsize,1, 44,44)) * sigmas[0]).cuda()
        start = xt.clone()
    eps = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps, eps_prev = model(xt.cuda(), sig.cuda(), visibility=condition, encoder_hidden_states=None, output_dict=False), eps
        eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * (torch.randn((batchsize,44,44))).unsqueeze(1).cuda()
    return xt, start

## Trained Models
sdf_generator = torch.load("...", weights_only = False).cuda()

vertices_generator = torch.load("...", weights_only = False).cuda()

sdf_generator.eval()
vertices_generator.eval()

from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
class FastVisibilityCalculator:
    def __init__(self, num_jobs=-1):
        self.num_jobs = num_jobs

    @staticmethod
    def segments_intersect_vectorized(p1, p2, edges):
        """Check if segment (p1, p2) intersects any segment in edges."""
        r = p2 - p1
        q1 = edges[:, 0]
        q2 = edges[:, 1]
        s = q2 - q1

        def cross(a, b):
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

        denom = cross(np.tile(r, (len(edges), 1)), s)
        zero_denom = np.abs(denom) < 1e-9

        t = cross(q1 - p1, s) / (denom + 1e-9)
        u = cross(q1 - p1, np.tile(r, (len(edges), 1))) / (denom + 1e-9)

        valid = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1) & (~zero_denom)
        return np.any(valid)

    @staticmethod
    def point_in_polygon(point, polygon):
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[(i + 1) % n]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
        return inside

    def process_row_vectorized(self, i, coords, edges):
        n = len(coords)
        vis_row = np.zeros(n)
        pi = coords[i]

        for j in range(i + 1, n):
            if j - i == 1 or (i == 0 and j == n - 1):
                continue
            pj = coords[j]

            # Filter out touching edges
            mask = ~((np.all(edges[:, 0] == pi, axis=1)) |
                     (np.all(edges[:, 1] == pi, axis=1)) |
                     (np.all(edges[:, 0] == pj, axis=1)) |
                     (np.all(edges[:, 1] == pj, axis=1)))

            if self.segments_intersect_vectorized(pi, pj, edges[mask]):
                continue

            midpoint = (pi + pj) / 2
            if not self.point_in_polygon(midpoint, coords):
                continue

            vis_row[j] = 1

        return vis_row

    def visibility_matrix_generator(self, coords):
        coords = np.asarray(coords)
        n = len(coords)
        visibility = np.zeros((n, n))

        np.fill_diagonal(visibility, 1)
        for i in range(n):
            visibility[i, (i + 1) % n] = 1
            visibility[(i + 1) % n, i] = 1

        edges = np.stack([coords, np.roll(coords, -1, axis=0)], axis=1)

        results = Parallel(n_jobs=self.num_jobs)(
            delayed(self.process_row_vectorized)(i, coords, edges) for i in range(n)
        )

        for i, row in enumerate(results):
            visibility[i, :] += row

        visibility = np.clip(visibility + visibility.T, 0, 1)
        return visibility

def sdf_polygon_generation(sdf):
    binary_mask = ((sdf > -0.1) & (sdf < 0.1)).astype(np.uint8) * 255
    polygon_contour, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(polygon_contour, key=cv2.contourArea)
    points_dis = np.array([pt[0] for pt in largest_contour], dtype=np.float32)
    simplifier = vw.Simplifier(points_dis)
    points_dis = simplifier.simplify(number=25)
    points_dis = points_dis * 0.05
    if points_dis.shape[0] < 25:
        points_dis = interpolate_to_n_points(points_dis)
    points_list = list()
    for shift in range(len(points_dis)):
        B_shift = np.roll(points_dis, shift, axis=0)
        B_shift_rev = np.roll(points_dis[::-1], shift, axis=0)
        points_list.append(B_shift)
        points_list.append(B_shift_rev)
    
    polygon_initialization = np.array(points_list)
    polygon_initialization = torch.tensor(polygon_initialization).to(torch.float32)
    return polygon_initialization

def interpolate_to_n_points(points, n=25):
    """
    Linearly interpolate a 2D point sequence to have exactly `n` points.
    
    Args:
        points (np.ndarray): shape [M, 2], original point sequence
        n (int): desired number of points (default is 25)

    Returns:
        np.ndarray: shape [n, 2], interpolated point sequence
    """
    if points.shape[0] == n:
        return points

    # Compute cumulative distances (arc length)
    distances = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    distances = np.insert(distances, 0, 0)

    # Target interpolation points
    interp_distances = np.linspace(0, distances[-1], n)

    # Interpolation function for x and y
    interp_func_x = interp1d(distances, points[:, 0], kind='linear')
    interp_func_y = interp1d(distances, points[:, 1], kind='linear')

    interpolated_points = np.stack([interp_func_x(interp_distances),
                                    interp_func_y(interp_distances)], axis=-1)
    return interpolated_points

schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
sigmas = schedule.sample_sigmas(50)

test_dataset = "/home/isleri/mahes092/Polygon_Dataset_Generator/Minimum_Feature_25_Dataset_Test_New/"
base_sigma = 0.05
sigma_increment = 0.05
max_depth = 5
children_per_node = 2
distance_threshold = 0.05
latent_shape = (44, 44)  # Or (C, 44, 44) if needed
no_samples = 1
existing_polygon_batch = list()

def align_polygon_to_first_edge(polygon):
    """
    Aligns the polygon such that the first edge is aligned with the x-axis
    and the first point is at the origin.

    Args:
        polygon (np.ndarray): shape (N, 2), list of (x, y) points in order.

    Returns:
        np.ndarray: aligned polygon of shape (N, 2)
    """
    p0 = polygon[0]
    p1 = polygon[1]
    edge_vec = p1 - p0

    # Compute angle to rotate first edge to x-axis
    angle = -np.arctan2(edge_vec[1], edge_vec[0])  # clockwise rotation

    # Rotation matrix
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    # Translate to origin and apply rotation
    aligned = (polygon - p0) @ R.T
    return aligned

def chamfer_distance(poly_a, poly_b):
    from scipy.spatial.distance import cdist
    dists_a_to_b = cdist(poly_a, poly_b)
    dists_b_to_a = cdist(poly_b, poly_a)
    return np.mean(np.min(dists_a_to_b, axis=1)) + np.mean(np.min(dists_b_to_a, axis=1))

def is_similar(new_poly, polygon_list, threshold=0.2):
    '''
    New Node Addition Polygon Similarity
    '''
    new_aligned = align_polygon_to_first_edge(new_poly)
    for poly in polygon_list:
        aligned_poly = align_polygon_to_first_edge(poly)
        dist = chamfer_distance(new_aligned, aligned_poly)
        if dist < threshold:
            return True, dist
    return False, None

def max_nodes(depth, children_per_node):
    '''
    Total Number of Nodes Possible
    '''
    if children_per_node == 1:
        return depth + 1
    return (children_per_node**(depth + 1) - 1) // (children_per_node - 1)

## BFS Based Coverage Metric Calculation
coverage_metric_list = list()
cnt = 0
for each_file in os.listdir(test_dataset):
    if os.path.isfile(test_dataset + each_file):
        with open(test_dataset +  each_file, 'rb') as f:
            polygon_dict = pickle.load(f)
        final_level_2_visibility = polygon_dict["level_2_visibility"]
        visibility = list()
        for each_entry in range(final_level_2_visibility.shape[0]):
            for each_entry_1 in range(each_entry, final_level_2_visibility.shape[1]):
                visibility.append(final_level_2_visibility[each_entry, each_entry_1])    
        actual_visibility = torch.tensor(visibility).to(torch.float32).unsqueeze(0).cuda()
        no_samples = 1
        visibility_encoded_batch = actual_visibility.repeat(no_samples, 1)
        
        queue = deque()
        root_noise = np.random.normal(0, base_sigma, size=latent_shape)
        queue.append((root_noise, 0))  # (noise_tensor, depth)

        # Step 3: BFS Traversal and Sampling
        processed_nodes = 0
        while queue:
            current_noise, depth = queue.popleft()
            sigma = base_sigma + depth * sigma_increment

            # Convert noise to torch
            noise_tensor = torch.tensor(current_noise).unsqueeze(0).unsqueeze(0).cuda().to(torch.float32)

            # Generate SDF
            generated_sdf_batch, start_noise = samples(
                sdf_generator,
                visibility_encoded_batch,
                sigmas,
                1.0,
                0.5,
                noise_tensor,
                no_samples
            )
            

            sdf_batch = generated_sdf_batch.squeeze(1).detach().cpu().numpy()
            final_vertices_generated_batch_all = list()
            for each_generated_sdf in range(generated_sdf_batch.shape[0]):
                current_sdf_polygon_list = list()
                current_sdf = sdf_batch[each_generated_sdf,:,:]
                points_initialization = sdf_polygon_generation(current_sdf)
                current_sdf_repeated = torch.tensor(current_sdf).unsqueeze(0).unsqueeze(0).repeat(points_initialization.shape[0], 1, 1, 1)
                visibility_encoded_vertices_batch = actual_visibility.repeat(current_sdf_repeated.shape[0], 1)
                vertices_generated_batch = vertices_generator(points_initialization.detach().cuda(), visibility_encoded_vertices_batch.detach().cuda(), current_sdf_repeated.detach().cuda())
                final_vertices_generated_batch = points_initialization.cuda() + vertices_generated_batch
                for each_ordering in range(final_vertices_generated_batch.shape[0]):
                    current_sdf_polygon_list.append(final_vertices_generated_batch[each_ordering].detach().cpu().numpy())
                final_vertices_generated_batch_all.append(current_sdf_polygon_list)

            final_vertices_generated_batch_all = np.array(final_vertices_generated_batch_all)
            final_generated_sdf_batch = generated_sdf_batch.detach().squeeze(1).cpu().numpy()
            final_vertices_generated_batch_all_filtered = list()
            for each_polygon in range(final_vertices_generated_batch_all.shape[0]):
                best_acc = None
                best_vertices_generated = None 
                best_visibility = None 
                best_sdf = None
                current_polygon_set = final_vertices_generated_batch_all[each_polygon]
                for each_sub_polygon in range(len(current_polygon_set)):
                    current_polygon = current_polygon_set[each_sub_polygon,:,:]
                    predicted_visibility = FastVisibilityCalculator().visibility_matrix_generator(current_polygon)
                    acc = f1_score(final_level_2_visibility.flatten(), predicted_visibility.flatten(), average = "macro")
                    if best_acc is not None:
                        if acc > best_acc:
                            best_vertices_generated = copy.deepcopy(current_polygon)
                            best_acc = copy.deepcopy(acc)
                            best_visibility = copy.deepcopy(predicted_visibility)
                            best_sdf = copy.deepcopy(final_generated_sdf_batch[each_polygon, :,:])
                    else:
                        best_vertices_generated = copy.deepcopy(current_polygon)
                        best_acc = copy.deepcopy(acc)
                        best_visibility = copy.deepcopy(predicted_visibility)
                        best_sdf = copy.deepcopy(final_generated_sdf_batch[each_polygon, :,:])
            if best_acc > 0.75:
                if depth < max_depth:
                    check_for_children = False
                    if len(existing_polygon_batch) == 0:
                        check_for_children = True 
                        processed_nodes += 1
                        existing_polygon_batch.append(best_vertices_generated)
                    else:
                        similar, dist = is_similar(best_vertices_generated, existing_polygon_batch, threshold=distance_threshold)
                        if not similar:
                            check_for_children = True 
                            processed_nodes += 1
                            existing_polygon_batch.append(best_vertices_generated)
                    if check_for_children:
                        for _ in range(children_per_node):
                            perturbation = np.random.normal(0, sigma, size=latent_shape)
                            child_noise = current_noise + perturbation
                            queue.append((child_noise, depth + 1))
        coverage_metric = processed_nodes/max_nodes(max_depth, children_per_node)
        coverage_metric_list.append(coverage_metric)
        cnt += 1
        print(cnt)
        if cnt % 10:
            print("Max Depth:", max_depth)
            print("Children Per Node:", children_per_node)
            print("Distance Threshold:", distance_threshold)
            print("Average Coverage Metric:", sum(coverage_metric_list)/len(coverage_metric_list))
print("Max Depth:", max_depth)
print("Children Per Node:", children_per_node)
print("Distance Threshold:", distance_threshold)
print("Average Coverage Metric:", sum(coverage_metric_list)/len(coverage_metric_list))



