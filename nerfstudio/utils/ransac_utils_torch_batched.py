# Copyright 2026 the authors of IDSplat and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

import torch


def compute_registration_fitness(
    points_src: torch.Tensor,
    points_tgt: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    threshold: float = 0.05,
    mode: str = "arithmetic",  # 'arithmetic', 'harmonic', or 'hybrid'
    alpha: float = 0.4,  # blend factor for hybrid mode
) -> float:
    """
    Compute a bidirectional geometric fitness score between two point clouds.

    Args:
        points_src: (N_src, 3) source cloud
        points_tgt: (N_tgt, 3) target cloud
        R, t: rigid transform (3x3, 3)
        threshold: inlier distance threshold
        mode: 'arithmetic' | 'harmonic' | 'hybrid'
        alpha: weight for hybrid mean (0 = pure arithmetic, 1 = pure harmonic)

    Returns:
        fitness: float in [0,1], higher = better alignment
    """

    points_src_trans = (R @ points_src.T).T + t.unsqueeze(0)  # (N_src,3)
    dists_src_tgt = torch.cdist(points_src_trans, points_tgt)  # (N_src, N_tgt)
    min_dists_src, _ = dists_src_tgt.min(dim=1)
    inlier_ratio_src = (min_dists_src < threshold).float().mean()

    points_tgt_trans_back = (R.T @ (points_tgt - t).T).T
    dists_tgt_src = torch.cdist(points_tgt_trans_back, points_src)
    min_dists_tgt, _ = dists_tgt_src.min(dim=1)
    inlier_ratio_tgt = (min_dists_tgt < threshold).float().mean()

    r_src = inlier_ratio_src.item()
    r_tgt = inlier_ratio_tgt.item()

    if mode == "arithmetic":
        fitness = 0.5 * (r_src + r_tgt)

    elif mode == "harmonic":
        eps = 1e-6  # to avoid divide-by-zero
        fitness = 2 * r_src * r_tgt / (r_src + r_tgt + eps)

    elif mode == "hybrid":
        eps = 1e-6
        f_arith = 0.5 * (r_src + r_tgt)
        f_harm = 2 * r_src * r_tgt / (r_src + r_tgt + eps)
        fitness = (1 - alpha) * f_arith + alpha * f_harm

    elif mode == "tgt_coverage":
        fitness = r_tgt

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'arithmetic', 'harmonic', or 'hybrid'.")

    return float(fitness)


def find_feature_matches_torch(
    features_src: torch.Tensor, features_tgt: torch.Tensor, mutual: bool = True, min_sim_for_match: float = 0.8
):
    """
    Simple GPU-friendly cosine matching (brute-force).
    Returns (N_matches, 2) int64 tensor of (i_src, j_tgt).
    """
    device = features_src.device
    f_src = torch.nn.functional.normalize(features_src, p=2, dim=1)
    f_tgt = torch.nn.functional.normalize(features_tgt, p=2, dim=1)

    # similarity matrix (N_src, N_tgt)
    sim = f_src @ f_tgt.t()

    best_src_vals, best_src_idx = sim.max(dim=0)
    best_src_idx = best_src_idx[best_src_vals > min_sim_for_match]
    corresponding_tgt_idx = torch.arange(features_tgt.size(0), device=device)[best_src_vals > min_sim_for_match]
    if not mutual:
        matches = torch.stack([best_src_idx, corresponding_tgt_idx], dim=1)
        return matches

    else:
        raise NotImplementedError("Mutual matching is not implemented yet.")
    return matches


def estimate_rigid_transform_batch(P: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Umeyama estimator.

    P, Q:
      - If performing single transform: shape (N,3)
      - If batched: P shape (B, N, 3), Q shape (B, N, 3) -> returns R (B,3,3), t (B,3)
    This implementation handles both cases by checking dims.
    """
    single = False
    if P.dim() == 2:
        single = True
        P = P.unsqueeze(0)  # (1, N, 3)
        Q = Q.unsqueeze(0)

    B, N, _ = P.shape

    centroid_P = P.mean(dim=1, keepdim=True)  # (B,1,3)
    centroid_Q = Q.mean(dim=1, keepdim=True)

    P_centered = P - centroid_P  # (B,N,3)
    Q_centered = Q - centroid_Q

    # compute covariance H = P_centered^T @ Q_centered for each batch
    # first transpose P_centered -> (B,3,N)
    H = P_centered.transpose(1, 2) @ Q_centered  # (B,3,3)

    # batched SVD
    # torch.linalg.svd returns U, S, Vh (Vh is V^T)
    U, S, Vh = torch.linalg.svd(H)  # U:(B,3,3), Vh:(B,3,3)
    V = Vh.transpose(-2, -1)  # (B,3,3)

    R = V @ U.transpose(-2, -1)  # (B,3,3)

    # ensure right-handed rotations: if det(R) < 0, flip last column of V
    detR = torch.det(R)
    if (detR < 0).any():
        # we must fix per-batch entry
        mask = detR < 0  # (B,)
        # flip Vh[last row] for masked batches
        Vh_fixed = Vh.clone()
        Vh_fixed[mask, -1, :] = -Vh_fixed[mask, -1, :]
        V_fixed = Vh_fixed.transpose(-2, -1)
        R = V_fixed @ U.transpose(-2, -1)

    # t = centroid_Q.squeeze(1) - R @ centroid_P.squeeze(1)
    centroid_P_s = centroid_P.squeeze(1)  # (B,3)
    centroid_Q_s = centroid_Q.squeeze(1)
    t = centroid_Q_s - (R @ centroid_P_s.unsqueeze(-1)).squeeze(-1)  # (B,3)

    if single:
        return R.squeeze(0), t.squeeze(0)
    return R, t


def compute_inliers_batch(P_all: torch.Tensor, Q_all: torch.Tensor, R: torch.Tensor, t: torch.Tensor, threshold: float):
    """
    Compute inliers for a batch of hypotheses.

    P_all: (M,3)
    Q_all: (M,3)
    R: (B,3,3)
    t: (B,3)
    Returns:
      - inlier_counts: (B,) long
      - inlier_masks: (B,M) bool  (optional heavy; but useful to extract indices)
    """
    # transform P_all by every R,t: produce (B, M, 3)
    # first compute R @ P_all.T -> (B,3,M)
    P_all_t = P_all.t().unsqueeze(0)  # (1,3,M)
    RP = R @ P_all_t  # (B,3,M)
    RP = RP.permute(0, 2, 1)  # (B,M,3)
    P_trans = RP + t.unsqueeze(1)  # (B,M,3)

    # compute distances to Q_all
    Q_all_unsq = Q_all.unsqueeze(0)  # (1,M,3)
    diffs = P_trans - Q_all_unsq  # (B,M,3)
    dists = torch.linalg.norm(diffs, dim=2)  # (B,M)
    inlier_mask = dists < threshold
    inlier_counts = inlier_mask.sum(dim=1)
    return inlier_counts, inlier_mask


def ransac_rigid_registration_torch_batched(
    points_src: torch.Tensor,
    points_tgt: torch.Tensor,
    features_src: torch.Tensor,
    features_tgt: torch.Tensor,
    num_hypotheses: int = 5000,
    batch_size: int = 256,
    distance_threshold: float = 0.05,
    mutual: bool = True,
    refine: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Batched/parallel RANSAC.

    - num_hypotheses: total number of 3-point hypotheses to evaluate
    - batch_size: how many hypotheses to evaluate in parallel per GPU batch (memory control)
    Returns (R_best, t_best, inlier_indices_of_best)
    """
    if device is None:
        device = points_src.device

    dtype = points_src.dtype if points_src.is_floating_point() else torch.float32
    points_src = points_src.to(device=device, dtype=dtype)
    points_tgt = points_tgt.to(device=device, dtype=dtype)
    features_src = features_src.to(device=device, dtype=dtype)
    features_tgt = features_tgt.to(device=device, dtype=dtype)

    matches = find_feature_matches_torch(features_src, features_tgt, mutual=mutual)
    n_matches = matches.shape[0]
    if n_matches < 3:
        print("Not enough correspondences found.")
        return None, None, torch.empty(0, dtype=torch.long, device=device)

    # Pre-extract correspondence point lists
    P_all = points_src[matches[:, 0]]  # (M,3)
    Q_all = points_tgt[matches[:, 1]]  # (M,3)

    best_inlier_count = -1
    best_R = None
    best_t = None
    best_inlier_mask = None

    hypotheses_done = 0
    while hypotheses_done < num_hypotheses:
        current_batch_size = min(batch_size, num_hypotheses - hypotheses_done)
        if n_matches >= 3:
            weights = torch.ones((current_batch_size, n_matches), device=device, dtype=dtype)
            triples = torch.multinomial(weights, num_samples=3, replacement=False)  # (current_batch_size,3)
        else:
            raise RuntimeError("n_matches < 3 (should have failed earlier)")

        # Create P_subsets and Q_subsets with shape (B, 3, 3)
        P_subset = points_src[matches[triples, 0]]
        Q_subset = points_tgt[matches[triples, 1]]

        # Estimate R,t per hypothesis (batched)
        R_batch, t_batch = estimate_rigid_transform_batch(P_subset, Q_subset)

        # Compute inliers for the whole match set for every hypothesis in the batch
        inlier_counts, inlier_masks = compute_inliers_batch(P_all, Q_all, R_batch, t_batch, distance_threshold)

        # Find best in this batch
        batch_best_count, batch_best_idx = torch.max(inlier_counts, dim=0)
        if batch_best_count.item() > best_inlier_count:
            best_inlier_count = int(batch_best_count.item())
            best_R = R_batch[batch_best_idx]
            best_t = t_batch[batch_best_idx]
            best_inlier_mask = inlier_masks[batch_best_idx]

        hypotheses_done += current_batch_size

    if best_R is None or best_inlier_mask is None or best_inlier_mask.sum() < 3:
        return None, None, torch.empty(0, dtype=torch.long, device=device)

    # Optional refinement on all inliers (single SVD on the inlier set)
    if refine:
        inlier_indices = torch.nonzero(best_inlier_mask, as_tuple=False).squeeze(1)
        P_in = P_all[inlier_indices]
        Q_in = Q_all[inlier_indices]
        R_refined, t_refined = estimate_rigid_transform_batch(P_in, Q_in)
        best_R, best_t = R_refined, t_refined

    final_inlier_indices = torch.nonzero(best_inlier_mask, as_tuple=False).squeeze(1)
    return best_R, best_t, final_inlier_indices


# Example usage & quick compare (not a rigorous benchmark)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.manual_seed(0)

    N = 2000  # number of points
    P = torch.rand(N, 3, device=device)
    theta = torch.tensor(30.0 * torch.pi / 180.0, device=device)
    R_true = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta), 0.0], [torch.sin(theta), torch.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    t_true = torch.tensor([0.5, 0.2, -0.3], device=device)
    Q = (R_true @ P.t()).t() + t_true.unsqueeze(0) + 0.005 * torch.randn_like(P)

    D = 64
    features_P = torch.randn(N, D, device=device)
    features_Q = features_P + 0.02 * torch.randn_like(features_P)

    # run the batched GPU RANSAC
    R_est, t_est, inliers = ransac_rigid_registration_torch_batched(
        P,
        Q,
        features_P,
        features_Q,
        num_hypotheses=4000,
        batch_size=512,
        distance_threshold=0.03,
        mutual=True,
        device=device,
        seed=42,
    )

    print("Estimated R:", R_est)
    print("Estimated t:", t_est)
    print("Inliers:", inliers.numel())
