import copy
import math
import numpy as np
import pickle
from itertools import tee
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from multiprocessing import Pool
from skimage.morphology import skeletonize, thin
import os
import seaborn as sns
import matplotlib.colors as mcolors
import cv2 
import visvalingamwyatt as vw
from scipy.interpolate import interp1d
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import Parallel, delayed
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
    xt = (torch.randn((batchsize,1, 44,44)) * sigmas[0]).cuda()
    eps = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps, eps_prev = model(xt.cuda(), sig.cuda(), visibility=condition, encoder_hidden_states=None, output_dict=False), eps
        eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * (torch.randn((batchsize,44,44))).unsqueeze(1).cuda()
    return xt

## Loading trained models
sdf_generator = torch.load("...", weights_only = False).cuda()

vertices_generator = torch.load("...", weights_only = False).cuda()


sdf_generator.eval()
vertices_generator.eval()

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
        

## General Dataset
actual_visibility_list = list()
predicted_visibility_list = list()
predicted_visibility_list_pre = list()

# Test Datset Path
test_dataset = "..."
i = 0
schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
sigmas = schedule.sample_sigmas(100)
evaluation_plots = {}

evaluation_plots_1 = {}
correct_count = 0
for each_file in os.listdir(test_dataset):
    if os.path.isfile(test_dataset + each_file):
        try:
            print(each_file)
            with open(test_dataset +  each_file, 'rb') as f:
                polygon_dict = pickle.load(f)
            final_level_2_visibility = polygon_dict["level_2_visibility"]
            actual_visibility = list()
            for each_entry in range(final_level_2_visibility.shape[0]):
                for each_entry_1 in range(each_entry, final_level_2_visibility.shape[1]):
                    actual_visibility.append(final_level_2_visibility[each_entry, each_entry_1]) 
            actual_visibility = torch.tensor(actual_visibility).unsqueeze(0).cuda().to(torch.float32)

            final_level_2_visibility_dummy = np.full_like(final_level_2_visibility, fill_value=-1)
            actual_visibility_dummy = list()
            for each_entry in range(final_level_2_visibility_dummy.shape[0]):
                for each_entry_1 in range(each_entry, final_level_2_visibility_dummy.shape[1]):
                    actual_visibility_dummy.append(final_level_2_visibility_dummy[each_entry, each_entry_1]) 
            actual_visibility_dummy = torch.tensor(actual_visibility_dummy).unsqueeze(0).cuda().to(torch.float32)
            no_samples = 50
            visibility_encoded_batch_dummy = actual_visibility_dummy.repeat(no_samples, 1)

            best_vertices_generated = None
            best_acc = None
            best_visibility = None
            best_sdf = None
            no_samples = 50
            visibility_encoded_batch = actual_visibility.repeat(no_samples, 1)

            

            generated_sdf_batch = samples(sdf_generator, visibility_encoded_batch_dummy, sigmas, 1.0, 0.5, None, no_samples)
            sdf_batch = generated_sdf_batch.squeeze(1).detach().cpu().numpy()
            final_vertices_generated_batch_all = list()
            for each_generated_sdf in range(generated_sdf_batch.shape[0]):
                current_sdf = sdf_batch[each_generated_sdf,:,:]
                points_initialization = sdf_polygon_generation(current_sdf)
                current_sdf_repeated = torch.tensor(current_sdf).unsqueeze(0).unsqueeze(0).repeat(points_initialization.shape[0], 1, 1, 1)
                visibility_encoded_vertices_batch = actual_visibility.repeat(current_sdf_repeated.shape[0], 1)
                vertices_generated_batch = vertices_generator(points_initialization.detach().cuda(), visibility_encoded_vertices_batch.detach().cuda(), current_sdf_repeated.detach().cuda())
                final_vertices_generated_batch = points_initialization.cuda() + vertices_generated_batch
                for each_ordering in range(final_vertices_generated_batch.shape[0]):
                    final_vertices_generated_batch_all.append(final_vertices_generated_batch[each_ordering].detach().cpu().numpy())
            final_vertices_generated_batch_all = np.array(final_vertices_generated_batch_all)
            final_generated_sdf_batch = generated_sdf_batch.detach().squeeze(1).cpu().numpy()
            for each_polygon in range(final_vertices_generated_batch_all.shape[0]):
                current_polygon = final_vertices_generated_batch_all[each_polygon,:]
                predicted_visibility = FastVisibilityCalculator().visibility_matrix_generator(current_polygon)
                acc = f1_score(final_level_2_visibility.flatten(), predicted_visibility.flatten(), average = "macro")
                if best_acc is not None:
                    if acc > best_acc:
                        best_vertices_generated = copy.deepcopy(current_polygon)
                        best_acc = copy.deepcopy(acc)
                        best_visibility = copy.deepcopy(predicted_visibility)
                        best_sdf = copy.deepcopy(final_generated_sdf_batch[each_polygon//50, :,:])
                else:
                    best_vertices_generated = copy.deepcopy(current_polygon)
                    best_acc = copy.deepcopy(acc)
                    best_visibility = copy.deepcopy(predicted_visibility)
                    best_sdf = copy.deepcopy(final_generated_sdf_batch[each_polygon//50, :,:])

            vertices_generated = copy.deepcopy(best_vertices_generated)
            acc = copy.deepcopy(best_acc)
            print(acc)
            print(np.sum(np.logical_and(best_visibility == 1, polygon_dict["level_2_visibility"] == 1))/np.sum(polygon_dict["level_2_visibility"] == 1) )
            print(np.sum(np.logical_and(best_visibility == 0, polygon_dict["level_2_visibility"] == 0))/np.sum(polygon_dict["level_2_visibility"] == 0) )
            predicted_visibility =  copy.deepcopy(best_visibility)

            actual_visibility_list.append(final_level_2_visibility)
            predicted_visibility_list_pre.append(predicted_visibility)

            opt_best_vertices = copy.deepcopy(vertices_generated)
            opt_best_acc = copy.deepcopy(acc)
            opt_best_visibility = copy.deepcopy(predicted_visibility)
            opt_pre_acc = copy.deepcopy(acc)
            opt_pre_vertices = copy.deepcopy(vertices_generated)

            best_optimization_acc = copy.deepcopy(opt_best_acc)
            predicted_visibility_list.append(opt_best_visibility)

            current_file = {}
            current_file["predicted_polygon"] = best_vertices_generated
            current_file["predicted_visibility"] = predicted_visibility
            current_file["true_polygon"] = polygon_dict["level_2_polygon"]
            current_file["true_visibility"] = polygon_dict["level_2_visibility"]
            current_file["predicted_sdf"] = best_sdf
            current_file["true_sdf"] = polygon_dict["small_continuous_sdf_image"]
            evaluation_plots[each_file] = current_file
        except:
            print("Error Processing Results")

## Saving the predictions for quantitative calulations
with open('Visdiff_Predictions_Visibility_Final_Ordering_Diffusion_Pixel_Aligned_Global_Unconditional.pkl', 'wb') as f:
    pickle.dump(evaluation_plots, f)