import torch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)  # This sets the seed for all GPUs
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import os
import numpy as np
import copy
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import pickle
import math
from torch.nn.utils.rnn import pad_sequence
import random
import matplotlib.pyplot as plt
import time
from unet_sdf_diffusion import *
import cv2
import joblib
random.seed(123)
np.random.seed(123)

def collate_fn(batch):
    polygon_batch  = []
    visibility_batch = []

    polygon_data = joblib.load(batch[0])
    for each_key in polygon_data.keys():
        ## SDF Normalization
        polygon = polygon_data[each_key]["small_continuous_sdf_image"]/np.max(polygon_data[each_key]["small_continuous_sdf_image"])
        ## This is for triangulation dual. Use the commented version for Visibility graph
        visibility_full = polygon_data[each_key]["triangulation_graph"]
        ## visibility_full = polygon_data[each_key]["visibility"]
        visibility = list()
        ## Getting the upper-triangular matrix
        for each_entry in range(visibility_full.shape[0]):
            for each_entry_1 in range(each_entry, visibility_full.shape[1]):
                visibility.append(visibility_full[each_entry, each_entry_1]) 

        polygon_batch.append(torch.tensor(polygon))
        visibility_batch.append(torch.tensor(visibility))

    polygon_batch = torch.stack(polygon_batch)
    visibility_batch = torch.stack(visibility_batch)
    return polygon_batch.to(torch.float32), visibility_batch.to(torch.float32)

print("Started Data Loader")
## Dataset Unzipped Folder: Change this to your path
dataset_path = "..."
data_list = list()
count = 0
for each in sorted(os.listdir(dataset_path)):
    if os.path.isfile(os.path.join(dataset_path, each)):
        data_list.append(os.path.join(dataset_path, each))

iterations = 200
data_loader = DataLoader(data_list,collate_fn=collate_fn, batch_size=1, shuffle=True, num_workers=12)
## Model Saving Folder: Change this to your path
experiment_folder = "..."

model = UNet_SD(in_channels= 1,
                base_channels=32,
                 time_emb_dim=512,
                 context_dim=512,
                 multipliers=(1, 2, 4),
                 attn_levels=(0,1),
                 nResAttn_block=1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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

epoch_number = 0
loss_curve = []
for epoch in range(epoch_number, iterations+1):
    running_loss_bce = 0.0
    running_loss = 0.0
    total_number_of_batches = 0
    model.train()
    print("Epoch:", epoch)
    for i, data in enumerate(data_loader, 0):
        polygon, level_2_visibility = data
        polygon, level_2_visibility  = polygon.cuda(non_blocking=True), level_2_visibility.cuda(non_blocking=True)
        optimizer.zero_grad()
        sigma, eps = generate_train_sample(polygon, schedule)
        noisy_input = (polygon + sigma.view(eps.shape[0], 1, 1) * eps).unsqueeze(1)
        eps_hat = model(noisy_input, sigma, visibility=level_2_visibility, encoder_hidden_states=None, output_dict=False)
        loss = nn.MSELoss()(eps_hat.squeeze(1), eps)
        loss.backward()
        optimizer.step()
        running_loss_bce += loss.item()
        total_number_of_batches = i
    loss_curve.append(running_loss / total_number_of_batches) ## to store the loss curve
    print(running_loss_bce / total_number_of_batches)
    with open( experiment_folder + 'training.log', 'a+') as f:
        f.write("Epoch:" + str(epoch) + " MSE Loss:" + str(running_loss_bce / total_number_of_batches) + "\n")

    if epoch % 10 == 0:
        torch.save(model, os.path.join(experiment_folder, 'sdf_generator_epoch-{}.pt'.format(epoch)))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(experiment_folder, 'sdf_generator_epoch_optimizer-{}.pt'.format(epoch)))


plt.plot(np.linspace(0, iterations, iterations+1), loss_curve[:(iterations+1)])
plt.title('Training Loss Curve')
plt.savefig(os.path.join(experiment_folder, 'Loss_Curve.png'))
