import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from Sequence_Prediction_Architecture_Pixel_Aligned import *
import torch.autograd as autograd
import joblib
import cv2 
import visvalingamwyatt as vw
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d
print("Library Loaded")
torch.manual_seed(124)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(124)  # This sets the seed for all GPUs

def interpolate_to_n_points(points, n=25):
    """
    Contour Extraction:

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

def collate_fn(batch):
    '''
    Batch Extraction
    '''
    level_2_visibility_batch  = []
    level_2_polygon_batch = []
    level_2_sdf_batch = []
    level_2_polygon_initial_batch = []

    polygon_data = joblib.load(batch[0])
            
    for each_key in polygon_data.keys():
        ## This is for triangulation dual. Use the commented version for Visibility graph
        visibility_full = polygon_data[each_key]["triangulation_graph"]
        ## visibility_full = polygon_data[each_key]["visibility"]
        visibility = list()
        ## Getting the upper-triangular matrix
        for each_entry in range(visibility_full.shape[0]):
            for each_entry_1 in range(each_entry, visibility_full.shape[1]):
                visibility.append(visibility_full[each_entry, each_entry_1]) 

        level_2_visibility_batch.append(torch.tensor(visibility))
        polygon_vertices = polygon_data[each_key]["level_2_polygon"]
        level_2_polygon_batch.append(torch.tensor(polygon_vertices))
        ## SDF Normalization
        level_2_sdf_batch.append(torch.tensor(polygon_data[each_key]["small_continuous_sdf_image"]/ np.max(np.abs(polygon_data[each_key]["small_continuous_sdf_image"]))))
    level_2_visibility_batch = torch.stack(level_2_visibility_batch)
    level_2_polygon_batch = torch.stack(level_2_polygon_batch)
    level_2_sdf_batch = torch.stack(level_2_sdf_batch).squeeze(1)
    return  level_2_visibility_batch.to(torch.float32), level_2_sdf_batch.to(torch.float32), level_2_polygon_batch.to(torch.float32)

class Schedule:
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas
    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]
    def __len__(self) -> int:
        return len(self.sigmas)
    def sample_batch(self, x0:torch.FloatTensor) -> torch.FloatTensor:
        return self[torch.randint(len(self), (x0.shape[0],))].to(x0)

def generate_train_sample(x0: torch.FloatTensor, schedule: Schedule):
    sigma = schedule.sample_batch(x0)
    eps = torch.randn_like(x0)
    return sigma, eps

class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float=10):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))

schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)

def generate_train_sample(x0: torch.FloatTensor, schedule: Schedule):
    sigma = schedule.sample_batch(x0)
    eps = torch.randn_like(x0)
    return sigma, eps


## Model Saving Folder: Change this to your path
experiment_folder = "..."
print("Started Data Loader")

## Dataset Unzipped Folder: Change this to your path
dataset_path = "..."
data_list = list()
for each in sorted(os.listdir(dataset_path)):
    if os.path.isfile(os.path.join(dataset_path, each)):
        data_list.append(os.path.join(dataset_path, each))
print(len(data_list))  
data_loader = DataLoader(data_list,collate_fn=collate_fn, batch_size=1, shuffle=True, num_workers=10)
model = PolygonVisibilityTransformerDecoderIntermediate().cuda()

## SDF Diffusion Trained Model
sdf_diffusion = torch.load("/home/isleri/mahes092/Polygon_Dataset_Generator/Continuous_SDF_Generation/Continuous_SDF/SDF_Diffusion_Normalized_Dual/sdf_generator_epoch-60.pt", weights_only = False).cuda()
sdf_diffusion.eval()
print(sum(p.numel() for p in sdf_diffusion.parameters() if p.requires_grad))
iterations = 200
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_curve = []
print("started")
import time
best_loss = None
def reorder_polygon(B, A):
    '''
    Polygon Ordering Check:

    Try all ordering of the polygons
    '''
    n = len(B)
    min_dist = float('inf')
    best_perm = None

    # Try all cyclic shifts
    for shift in range(n):
        B_shift = np.roll(B, shift, axis=0)
        dist = np.linalg.norm(A - B_shift)
        if dist < min_dist:
            min_dist = dist
            best_perm = B_shift

        # Try reversed (reflected) version as well
        B_shift_rev = np.roll(B[::-1], shift, axis=0)
        dist_rev = np.linalg.norm(A - B_shift_rev)
        if dist_rev < min_dist:
            min_dist = dist_rev
            best_perm = B_shift_rev

    return best_perm

def subsample_uniform(points, n=25):
    """
    Polygon Contour helper function

    Subsample `n` uniformly spaced points from an ordered polyline.

    Args:
        points (np.ndarray): shape (m, 2) with m >= n
        n (int): number of points to sample (n <= m)

    Returns:
        np.ndarray: shape (n, 2), uniformly sampled points
    """
    m = points.shape[0]
    if n >= m:
        return points  # or interpolate instead
    indices = np.sort(np.linspace(0, m - 1, 25, dtype=int))
    return points[indices]
    
def extract_contours_ordering(sdf, gt_points):
    '''
    This Function extracts the contour and aligns the contour with
    the GT ordering
    '''
    polygon_initalization = list()
    ground_truth = gt_points
    for each_sdf in range(sdf.shape[0]):
        current_sdf = sdf[each_sdf,:,:]
        current_gt = ground_truth[each_sdf,:,:]
        binary_mask = ((current_sdf > -0.1) & (current_sdf < 0.1)).astype(np.uint8) * 255
        points_dis = list()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            ## Getting the contour with largest area
            largest_contour = max(contours, key=cv2.contourArea)
            points_dis = np.array([pt[0] for pt in largest_contour], dtype=np.float32)
            if points_dis.shape[0] == 1:
                points_dis = np.tile(points_dis[0], (25, 1))
            if points_dis.shape[0] < 25:
                points_dis = interpolate_to_n_points(points_dis)
            elif points_dis.shape[0] > 25:
                simplifier = vw.Simplifier(points_dis)
                points_dis = simplifier.simplify(number=25)
        else:
            points_dis = np.zeros((25, 2), dtype=np.float32)
        points_dis = points_dis * 0.05
        points_dis = reorder_polygon(points_dis, current_gt)
        polygon_initalization.append(points_dis)
    return np.array(polygon_initalization)
        
for epoch in range(iterations+1):
    running_loss = 0.0
    running_crossing_loss = 0.0
    running_visibility_loss = 0.0
    running_mse_loss = 0.0
    running_sdf_loss = 0.0
    total_number_of_batches = 0
    model.train()
    print("Epoch:", epoch)
    for i, data in enumerate(data_loader, 0):
        level_2_visibility, level_2_sdf, level_2_polygon = data
        level_2_visibility, level_2_sdf, level_2_polygon = level_2_visibility.cuda(non_blocking=True), level_2_sdf.cuda(non_blocking=True), level_2_polygon.cuda(non_blocking=True)
        optimizer.zero_grad()

        sigma, eps = generate_train_sample(level_2_sdf, schedule)
        noisy_input = (level_2_sdf + sigma.view(eps.shape[0], 1, 1) * eps).unsqueeze(1)
        eps_hat = sdf_diffusion(noisy_input, sigma, visibility=level_2_visibility, encoder_hidden_states=None, output_dict=False).detach()
        predicted_sdf = (noisy_input.squeeze(1) - sigma.view(eps.shape[0], 1, 1) * eps_hat.squeeze(1))

        pred_sdf = predicted_sdf.detach().cpu().numpy()
        gt = level_2_polygon.detach().cpu().numpy()
        points_initial = torch.tensor(extract_contours_ordering(pred_sdf, gt)).cuda().to(torch.float32)
        pred = model(points_initial, level_2_visibility, predicted_sdf.unsqueeze(1))
        final_pred_coords = points_initial + pred
        loss_coor = nn.MSELoss()(final_pred_coords, level_2_polygon)
        loss = loss_coor
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_mse_loss += loss_coor.item()
        total_number_of_batches = i

    loss_curve.append(running_loss / total_number_of_batches) ## to store the loss curve
    with open( experiment_folder + 'training.log', 'a+') as f:
        f.write("Epoch:" + str(epoch) + " Loss:" + str(running_loss / total_number_of_batches) + " Visibility Loss:" + str(running_visibility_loss / total_number_of_batches) + " MSE Loss:" + str(running_mse_loss / total_number_of_batches) + " SDF Loss:" + str(running_sdf_loss / total_number_of_batches)+ " Crossing Loss:" + str(running_crossing_loss / total_number_of_batches) + "\n")

    if best_loss is None:
        torch.save(model, os.path.join(experiment_folder, 'sdf_vertices_generator_best.pt'.format(epoch)))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(experiment_folder, 'sdf_vertices_generator_epoch_optimizer_best.pt'.format(epoch)))
        best_loss = loss.item()
    elif loss.item() < best_loss:
        torch.save(model, os.path.join(experiment_folder, 'sdf_vertices_generator_best.pt'.format(epoch)))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(experiment_folder, 'sdf_vertices_generator_epoch_optimizer_best.pt'.format(epoch)))
        best_loss = loss.item()


plt.plot(np.linspace(0, iterations, iterations+1), loss_curve[:(iterations+1)])
plt.title('Training Loss Curve')
plt.savefig(os.path.join(experiment_folder, 'Loss_Curve.png'))
