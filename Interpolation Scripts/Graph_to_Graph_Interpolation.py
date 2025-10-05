#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisDiff triangulation propagation — 3-panel GIF (constant middle box size)

Left:  Start polygon
Middle: Generated polygon per flip (updates each frame)
Right: End polygon

Plot style matches your snippet:
- blue edges, linewidth=10
- viridis-colored vertices with markersize=20
- axes off, equal aspect, tight layout

Dependencies:
  pip install torch numpy matplotlib shapely opencv-python visvalingamwyatt
"""

# ------------------------- Imports -------------------------
import os
import math
import copy
import pickle
import numpy as np
from itertools import tee

import torch
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
from matplotlib import animation

from shapely.geometry import Polygon
import cv2
import visvalingamwyatt as vw

from Edge_Flip_triangulation import transform_triangulation

# ------------------------- Repro & Device -------------------------
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Diffusion Schedule -------------------------
class Schedule:
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas
    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]
    def __len__(self) -> int:
        return len(self.sigmas)
    def sample_batch(self, x0: torch.FloatTensor) -> torch.FloatTensor:
        return self[torch.randint(len(self), (x0.shape[0],))].to(x0.device)
    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
        indices = list((len(self) * (1 - np.arange(0, steps)/steps)).round().astype(np.int64) - 1)
        return self[indices + [0]]

class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float = 0.02, sigma_max: float = 10.0):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# ------------------------- Diffusion Sampler -------------------------
@torch.no_grad()
def samples(model, condition, sigmas, gam, mu, xt, batchsize):
    """
    Uses your model interface: model(xt, sig, visibility=condition, ...)
    Returns final xt (SDF) and start noise.
    """
    start = None
    if xt is None:
        xt = (torch.randn((batchsize, 1, 44, 44), device=DEVICE) * sigmas[0]).to(DEVICE)
        start = xt.clone()
    eps_prev = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps = model(xt, sig.to(DEVICE), visibility=condition, encoder_hidden_states=None, output_dict=False)
        eps_av = eps if (i == 0 or eps_prev is None) else (eps * gam + eps_prev * (1 - gam))
        eps_prev = eps
        # DDIM-ish step
        sig_p = (sig_prev / (sig ** mu)) ** (1.0 / (1.0 - mu))
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * torch.randn_like(xt)
    return xt, start

# ------------------------- Utilities -------------------------
def pack_upper_tri(M: np.ndarray) -> torch.Tensor:
    vals = []
    n = M.shape[0]
    for r in range(n):
        for c in range(r, n):
            vals.append(M[r, c])
    return torch.tensor(vals, dtype=torch.float32, device=DEVICE).unsqueeze(0)

def _interp_to_n(points: np.ndarray, n: int = 25) -> np.ndarray:
    if points.shape[0] == n:
        return points
    d = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    d = np.insert(d, 0, 0.0)
    t = np.linspace(0.0, d[-1], n)
    x = np.interp(t, d, points[:, 0])
    y = np.interp(t, d, points[:, 1])
    return np.stack([x, y], axis=-1)

def sdf_polygon_generation(sdf: np.ndarray, band: float = 0.1, target_n: int = 25, scale: float = 0.05) -> torch.Tensor:
    """
    From a single SDF (H,W) numpy array:
      - extract iso-band |sdf|<band
      - largest contour -> simplify to ~target_n with Visvalingam–Whyatt
      - resample to exactly target_n if needed
      - generate multiple orderings: all rotations and reversed rotations
    Returns: torch.FloatTensor [M, target_n, 2]
    """
    mask = ((sdf > -band) & (sdf < band)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return torch.zeros((1, target_n, 2), dtype=torch.float32)

    largest = max(contours, key=cv2.contourArea)
    pts = np.array([p[0] for p in largest], dtype=np.float32)
    simp = vw.Simplifier(pts).simplify(number=target_n)
    simp = simp * scale
    if simp.shape[0] < target_n:
        simp = _interp_to_n(simp, n=target_n)

    cands = []
    for k in range(len(simp)):
        cands.append(np.roll(simp, k, axis=0))
        cands.append(np.roll(simp[::-1], k, axis=0))
    return torch.tensor(np.stack(cands, axis=0), dtype=torch.float32)

@torch.no_grad()
def vertices_from_sdf(vertices_generator, sdf_t: torch.Tensor, cond_tensor: torch.Tensor) -> np.ndarray:
    """
    Build candidate initializations from SDF → run vertex head (predicts offsets) → choose first valid polygon.
    Args:
      sdf_t: (1,1,H,W) torch
      cond_tensor: (1,D)
    Returns:
      (V,2) numpy array (chosen polygon). Falls back to first if none valid.
    """
    sdf_np = sdf_t.detach().squeeze().cpu().numpy()  # (H,W)
    init_pts = sdf_polygon_generation(sdf_np)        # (M,V,2)
    M, V, _ = init_pts.shape

    sdf_rep  = torch.tensor(sdf_np, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0).repeat(M, 1, 1, 1)
    cond_rep = cond_tensor.to(DEVICE).repeat(M, 1)
    init_pts = init_pts.to(DEVICE)

    offs = vertices_generator(init_pts, cond_rep, sdf_rep)   # (M,V,2)
    final_batch = (init_pts + offs).detach().cpu().numpy()   # (M,V,2)

    for b in range(M):
        pts = final_batch[b]
        try:
            poly = Polygon(pts)
            if poly.is_valid and not poly.is_empty and poly.area > 1e-8:
                return pts
        except Exception:
            pass
    return final_batch[0]

# ---------------------- Best valid (else best) vertex recovery ----------------------
# @torch.no_grad()
# def vertices_from_sdf(vertices_generator, sdf_t: torch.Tensor, cond_tensor: torch.Tensor) -> np.ndarray:
#     """
#     Generate multiple candidate polygons from an SDF by running the vertex head.
#     Select the best *valid* polygon by F1 vs the target visibility.
#     If none valid, pick the best overall; else fall back to the first.
#     """
#     from sklearn.metrics import f1_score

#     def unpack_visibility(cond_vec: torch.Tensor, V: int) -> np.ndarray:
#         u = cond_vec.detach().float().view(-1).cpu().numpy()
#         M = np.zeros((V, V), dtype=np.float32)
#         idx = 0
#         for r in range(V):
#             for c in range(r, V):
#                 M[r, c] = u[idx]
#                 M[c, r] = u[idx]
#                 idx += 1
#         np.fill_diagonal(M, 1.0)
#         return (M >= 0.5).astype(np.float32)

#     sdf_np = sdf_t.detach().squeeze().cpu().numpy()      # (H,W)
#     init_pts = sdf_polygon_generation(sdf_np)             # (M,V,2)
#     M, V, _ = init_pts.shape

#     sdf_rep  = torch.tensor(sdf_np, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0).repeat(M, 1, 1, 1)
#     cond_rep = cond_tensor.to(DEVICE).repeat(M, 1)
#     init_pts = init_pts.to(DEVICE)

#     offs = vertices_generator(init_pts, cond_rep, sdf_rep)   # (M,V,2) offsets
#     finals = (init_pts + offs).detach().cpu().numpy()        # (M,V,2)

#     target_vis = unpack_visibility(cond_tensor, V)
#     vis_calc = Visibility_Calculator()

#     best_valid_idx, best_valid_score = None, -np.inf
#     best_any_idx, best_any_score = None, -np.inf

#     for b in range(M):
#         pts = finals[b]
#         score = -np.inf
#         is_valid = False
#         try:
#             poly = Polygon(pts)
#             is_valid = (poly.is_valid and (not poly.is_empty) and (poly.area > 1e-8))
#             pred_vis = vis_calc.visibility_matrix_generator(pts)
#             score = f1_score(target_vis.flatten(), pred_vis.flatten(), average="macro")
#         except Exception:
#             pass

#         if score > best_any_score:
#             best_any_score, best_any_idx = score, b
#         if is_valid and score > best_valid_score:
#             best_valid_score, best_valid_idx = score, b

#     if best_valid_idx is not None:
#         return finals[best_valid_idx]
#     if best_any_idx is not None:
#         return finals[best_any_idx]
#     return finals[0]

# ------------------------- Bounds (keep box size constant) -------------------------
def compute_global_square_bounds(polys, pad_ratio=0.06):
    """
    Compute a square bounding box that contains all polygons, with padding.
    Returns (xmin, xmax, ymin, ymax).
    """
    pts_all = [p for p in polys if p is not None and len(p) > 0]
    if not pts_all:
        return (-1, 1, -1, 1)
    stack = np.vstack(pts_all)
    minx, miny = stack.min(0)
    maxx, maxy = stack.max(0)
    dx = maxx - minx
    dy = maxy - miny
    side = max(dx, dy)
    pad = pad_ratio * (side if side > 0 else 1.0)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    half = 0.5 * side + pad
    return (cx - half, cx + half, cy - half, cy + half)

# ------------------------- Three-Panel Animator (constant axes) -------------------------
def animate_start_middle_end(
    start_polygon,
    generated_polygons,     # list of (V,2) arrays, one per flip/frame
    end_polygon,
    flip_indices=None,      # ignored now
    out_gif="Visdiff_Triangulation_Propagation_3panel.gif",
    fps=2,
    figsize=(24, 6)
):
    """GIF: [Start | Generated(step k) | End] with CONSTANT axes and a single step label."""
    assert len(generated_polygons) > 0, "generated_polygons is empty."

    # ---- constant axes bounds across all frames ----
    def compute_global_square_bounds(polys, pad_ratio=0.06):
        pts_all = [p for p in polys if p is not None and len(p) > 0]
        if not pts_all:
            return (-1, 1, -1, 1)
        stack = np.vstack(pts_all)
        minx, miny = stack.min(0)
        maxx, maxy = stack.max(0)
        dx = maxx - minx; dy = maxy - miny
        side = max(dx, dy)
        pad = pad_ratio * (side if side > 0 else 1.0)
        cx = (minx + maxx) / 2.0; cy = (miny + maxy) / 2.0
        half = 0.5 * side + pad
        return (cx - half, cx + half, cy - half, cy + half)

    xmin, xmax, ymin, ymax = compute_global_square_bounds(
        [start_polygon, end_polygon] + generated_polygons, pad_ratio=0.06
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable='box')
        ax.axis('off')

    # Leave extra headroom; put the step label above the plots
    fig.subplots_adjust(top=0.86, wspace=0.10)
    step_text = fig.text(0.5, 0.93, "", ha="center", va="top", fontsize=28, fontweight="bold")

    # --- plotting style (matches your snippet) ---
    def _plot_polygon_style(ax, polygon, title=None):
        ax.cla()
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        if polygon is not None and len(polygon) > 0:
            colormap = plt.cm.get_cmap('viridis', len(polygon))
            ax.plot(polygon[:,0], polygon[:,1], color="blue", linewidth=10)
            ax.plot([polygon[-1,0], polygon[0,0]],
                    [polygon[-1,1], polygon[0,1]],
                    color="blue", linewidth=10)
            for i, (x, y) in enumerate(polygon):
                ax.plot(x, y, marker="o", color=colormap(i), markersize=20)
        if title is not None:
            ax.set_title(title, fontsize=24, weight="bold")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')

    def init():
        _plot_polygon_style(axes[0], start_polygon, title="Start")
        _plot_polygon_style(axes[2], end_polygon,   title="End")
        _plot_polygon_style(axes[1], generated_polygons[0], title=None)  # no middle title
        step_text.set_text(f"Step 1/{len(generated_polygons)}")
        return []

    def update(k):
        _plot_polygon_style(axes[0], start_polygon, title="Start")
        _plot_polygon_style(axes[2], end_polygon,   title="End")
        _plot_polygon_style(axes[1], generated_polygons[k], title=None)  # no middle title
        step_text.set_text(f"Step {k+1}/{len(generated_polygons)}")
        return []

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(generated_polygons), interval=1000/fps, blit=False
    )
    anim.save(out_gif, writer=animation.PillowWriter(fps=fps))
    print(f"Saved GIF: {out_gif}")
    plt.close(fig)


# ------------------------- Load Models -------------------------
SDF_CKPT  = "/projects/standard/isleri/mahes092/Polygon_Dataset_Generator/Models/SDF_Diffusion_Normalized_Dual/sdf_generator_epoch-60.pt"
VERT_CKPT = "/projects/standard/isleri/mahes092/Polygon_Dataset_Generator/Models/Vertex_Trained_Model_Dual/sdf_vertices_generator_best.pt"

print("Loading models...")
sdf_generator = torch.load(SDF_CKPT,  weights_only=False, map_location=DEVICE).to(DEVICE).eval()
vertices_generator = torch.load(VERT_CKPT, weights_only=False, map_location=DEVICE).to(DEVICE).eval()
print("Models loaded.")

# ------------------------- Main -------------------------
def main():
    # Paths defining start/end polygons
    file_1 = "/projects/standard/isleri/mahes092/Polygon_Dataset_Generator/Minimum_Feature_25_Dataset_Test_New/25_146_coords.pkl"
    file_2 = "/projects/standard/isleri/mahes092/Polygon_Dataset_Generator/Minimum_Feature_25_Dataset_Test_New/25_1_coords.pkl"

    with open(file_1, "rb") as f:
        polygon_dict = pickle.load(f)
    with open(file_2, "rb") as f:
        polygon_dict_1 = pickle.load(f)

    start_polygon = polygon_dict["level_2_polygon"]
    end_polygon   = polygon_dict_1["level_2_polygon"]

    # Triangulation sequence (gives intermediate visibility graphs)
    last_triangulation, flips, T_goal, T_start, polygon_start_triangulation, polygon_end_triangulation = transform_triangulation(
        start_polygon, end_polygon
    )
    print("Total flips:", len(flips))

    # Select a subset of flips for visualization
    number_of_intermediate_steps = 20
    idxs = np.linspace(0, len(flips) - 1, number_of_intermediate_steps, dtype=int)
    selected = [int(i) for i in idxs]
    print("Selected flip indices:", selected)

    # Diffusion schedule
    schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10.0)
    sigmas = schedule.sample_sigmas(50).to(DEVICE)

    generated_polygons = []
    kept_flip_ids = []

    for flip_idx in selected:
        vis_mat = flips[flip_idx]          # (V,V) numpy array
        cond    = pack_upper_tri(vis_mat)  # (1,D)

        # Sample SDF
        sdf_t, _ = samples(
            model=sdf_generator,
            condition=cond,
            sigmas=sigmas,
            gam=1.0,
            mu=0.5,
            xt=None,
            batchsize=1
        )  # (1,1,44,44)

        # Convert SDF -> polygon (init + vertex offsets)
        poly = vertices_from_sdf(vertices_generator, sdf_t, cond)   # (V,2)
        generated_polygons.append(poly)
        kept_flip_ids.append(flip_idx)
        print(f"Collected step for flip #{flip_idx} (valid={Polygon(poly).is_valid})")

    # Make 3-panel GIF with constant box size
    animate_start_middle_end(
        start_polygon=start_polygon,
        generated_polygons=generated_polygons,
        end_polygon=end_polygon,
        flip_indices=kept_flip_ids,
        out_gif="Visdiff_Triangulation_Propagation_3panel.gif",
        fps=2,
        figsize=(24, 8)
    )

if __name__ == "__main__":
    main()
