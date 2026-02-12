import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from tslearn.metrics import cdist_dtw
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

# Paths
FIGURE_PATH = '/data/gusev/USERS/jpconnor/figures/clinical_text_embedding_project/model_metrics/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/VTE_data/processed_datasets/')
STAGE_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/'
OUTPUT_PATH = os.path.join(RESULTS_PATH, 'phecode_model_comps_final/')
HELD_OUT_PRED_PATH = os.path.join(RESULTS_PATH, 'mortality_trajectories/')

trajectory_predictions_df = pd.read_csv(os.path.join(HELD_OUT_PRED_PATH, 'held_out_preds_full_cohort.csv'))
monthly_cols = [col for col in trajectory_predictions_df if col != 'DFCI_MRN']

missing_data_colwise = trajectory_predictions_df[monthly_cols].isna().sum(axis=0)
months_to_keep = missing_data_colwise[missing_data_colwise < 9 * len(trajectory_predictions_df) / 10].index

missing_data_rowise = (~trajectory_predictions_df[monthly_cols].isna()).sum(axis=1)
pxs_to_keep = missing_data_rowise[missing_data_rowise > 18].index

px_trajectories = trajectory_predictions_df.loc[pxs_to_keep, months_to_keep].values

n_patients, n_timepoints = px_trajectories.shape
time = np.arange(n_timepoints)

## Plot sample trajectories
slopes = np.full(n_patients, np.nan)
first_half_trend = np.full(n_patients, np.nan)
second_half_trend = np.full(n_patients, np.nan)

# Compute slopes and half-trends
half = n_timepoints // 2
for i in tqdm(list(range(n_patients))):
    y = px_trajectories[i]
    observed = ~np.isnan(y)
    if observed.sum() > 1:
        coeffs = np.polyfit(time[observed], y[observed], 1)
        slopes[i] = coeffs[0]

        # slope in first half
        obs_first = observed & (time < half)
        if obs_first.sum() > 1:
            first_half_trend[i] = np.polyfit(time[obs_first], y[obs_first], 1)[0]
        # slope in second half
        obs_second = observed & (time >= half)
        if obs_second.sum() > 1:
            second_half_trend[i] = np.polyfit(time[obs_second], y[obs_second], 1)[0]

# Define categories
increasing_idx = np.where(slopes > 0.01)[0]
decreasing_idx = np.where(slopes < -0.01)[0]
stable_idx     = np.where(np.abs(slopes) <= 0.01)[0]

# Nonlinear: slope changes sign between first and second half
nonlinear_idx = np.where((first_half_trend * second_half_trend) < 0)[0]

# Sample one from each category
np.random.seed(42)
sample_idx = [
    np.random.choice(increasing_idx) if len(increasing_idx) > 0 else None,
    np.random.choice(decreasing_idx) if len(decreasing_idx) > 0 else None,
    np.random.choice(stable_idx)     if len(stable_idx) > 0 else None,
    np.random.choice(nonlinear_idx)  if len(nonlinear_idx) > 0 else None
]

labels = ["Increasing", "Decreasing", "Stable", "Nonlinear"]

plt.figure(figsize=(14,8))
for idx, label in zip(sample_idx, labels):
    if idx is not None:
        plt.plot(time, px_trajectories[idx], marker='o', alpha=0.7, label=f"{label} (Patient {idx})")

plt.xlabel("Timepoint")
plt.ylabel("Risk Score")
plt.title("Sample Patient Trajectories by Trend Type")
plt.legend()

plt.savefig(FIGURE_PATH + 'sample_trajectories.png', dpi=300, bbox_inches='tight')
plt.close()

## Feature-based clustering
features = []

for row in tqdm(px_trajectories):
    observed = ~np.isnan(row)
    y_obs = row[observed]
    t_obs = time[observed]

    if len(y_obs) < 2:
        # update the number of NaNs to match number of features
        features.append([np.nan] * 12)
        continue

    # 1. Basic global features
    start_risk = y_obs[0]
    end_risk = y_obs[-1]
    mean_slope = (y_obs[-1] - y_obs[0]) / len(y_obs)
    auc = np.trapz(y_obs, t_obs) / len(y_obs)

    # 2. Early / mid / late averages
    third = len(y_obs) // 3
    early_mean = np.nanmean(y_obs[:third]) if third > 0 else np.nan
    mid_mean   = np.nanmean(y_obs[third:2*third]) if third > 0 else np.nan
    late_mean  = np.nanmean(y_obs[2*third:]) if third > 0 else np.nan

    # 3. Dynamic features
    
    # Minimum value (best response)
    min_val = np.nanmin(y_obs)
    time_to_min = t_obs[np.nanargmin(y_obs)]

    # Maximum value (worst point)
    max_val = np.nanmax(y_obs)
    time_to_max = t_obs[np.nanargmax(y_obs)]

    # Rebound: increase after minimum
    rebound = end_risk - min_val

    # Early slope (first half) vs late slope (second half)
    half = len(y_obs) // 2
    early_slope = (y_obs[half-1] - y_obs[0]) / half if half > 1 else np.nan
    late_slope = (y_obs[-1] - y_obs[half]) / (len(y_obs) - half) if len(y_obs) - half > 1 else np.nan

    # 4. Collect feature vector
    features.append([
        start_risk, end_risk, mean_slope, auc,
        early_mean, mid_mean, late_mean,
        min_val, time_to_min, rebound,
        early_slope, late_slope
    ])

features = np.array(features)
features_scaled = StandardScaler().fit_transform(features)

# Mask invalid patients
valid_mask = ~np.isnan(features).any(axis=1)
features_valid = features_scaled[valid_mask]

for n_clusters in range(2, 11):

    # Hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_labels_feat = np.full(n_patients, -1)
    cluster_labels_feat[valid_mask] = hc.fit_predict(features_valid)

    # Visualize cluster trajectories
    plt.figure(figsize=(14,8))
    for cl in range(n_clusters):
        members = px_trajectories[cluster_labels_feat == cl]
        if members.shape[0] == 0:
            continue
        mean_traj = np.nanmean(members, axis=0)
        lower = np.nanpercentile(members, 2.5, axis=0)
        upper = np.nanpercentile(members, 97.5, axis=0)
        plt.plot(time, mean_traj, label=f"Cluster {cl+1} mean", linewidth=3)
        plt.fill_between(time, lower, upper, alpha=0.2)
    plt.xlabel("Timepoint")
    plt.ylabel("Risk Score")
    plt.title(f'Feature-based Clustering using Hierarchical (n_clust={n_clusters})')
    plt.legend()

    plt.savefig(FIGURE_PATH + f"raw_trajectories/feature_clustering_hierarchical_{n_clusters}_clusters.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    
    
# (px_trajectories - np.nanmean(px_trajectories, axis=1).reshape(-1,1)) / np.nanstd(px_trajectories, axis=1).reshape(-1,1)
means = np.nanmean(px_trajectories, axis=1, keepdims=True)
stds = np.nanstd(px_trajectories, axis=1, keepdims=True)

# Avoid divide-by-zero (constant rows)
stds[stds == 0] = np.nan

scaled_px_trajectories = (px_trajectories - means) / stds

## Feature-based clustering
features = []

for row in tqdm(scaled_px_trajectories):
    observed = ~np.isnan(row)
    y_obs = row[observed]
    t_obs = time[observed]

    if len(y_obs) < 2:
        # update the number of NaNs to match number of features
        features.append([np.nan] * 12)
        continue

    # 1. Basic global features
    start_risk = y_obs[0]
    end_risk = y_obs[-1]
    mean_slope = (y_obs[-1] - y_obs[0]) / len(y_obs)
    auc = np.trapz(y_obs, t_obs) / len(y_obs)

    # 2. Early / mid / late averages
    third = len(y_obs) // 3
    early_mean = np.nanmean(y_obs[:third]) if third > 0 else np.nan
    mid_mean   = np.nanmean(y_obs[third:2*third]) if third > 0 else np.nan
    late_mean  = np.nanmean(y_obs[2*third:]) if third > 0 else np.nan

    # 3. Dynamic features
    
    # Minimum value (best response)
    min_val = np.nanmin(y_obs)
    time_to_min = t_obs[np.nanargmin(y_obs)]

    # Maximum value (worst point)
    max_val = np.nanmax(y_obs)
    time_to_max = t_obs[np.nanargmax(y_obs)]

    # Rebound: increase after minimum
    rebound = end_risk - min_val

    # Early slope (first half) vs late slope (second half)
    half = len(y_obs) // 2
    early_slope = (y_obs[half-1] - y_obs[0]) / half if half > 1 else np.nan
    late_slope = (y_obs[-1] - y_obs[half]) / (len(y_obs) - half) if len(y_obs) - half > 1 else np.nan

    # 4. Collect feature vector
    features.append([
        start_risk, end_risk, mean_slope, auc,
        early_mean, mid_mean, late_mean,
        min_val, time_to_min, rebound,
        early_slope, late_slope
    ])

features = np.array(features)
features_scaled = StandardScaler().fit_transform(features)

# Mask invalid patients
valid_mask = ~np.isnan(features).any(axis=1)
features_valid = features_scaled[valid_mask]

for n_clusters in range(2, 11):

    # Hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_labels_feat = np.full(n_patients, -1)
    cluster_labels_feat[valid_mask] = hc.fit_predict(features_valid)

    # Visualize cluster trajectories
    plt.figure(figsize=(14,8))
    for cl in range(n_clusters):
        members = scaled_px_trajectories[cluster_labels_feat == cl]
        if members.shape[0] == 0:
            continue
        mean_traj = np.nanmean(members, axis=0)
        lower = np.nanpercentile(members, 2.5, axis=0)
        upper = np.nanpercentile(members, 97.5, axis=0)
        plt.plot(time, mean_traj, label=f"Cluster {cl+1} mean", linewidth=3)
        plt.fill_between(time, lower, upper, alpha=0.2)
    plt.xlabel("Timepoint")
    plt.ylabel("Risk Score")
    plt.title(f'Feature-based Clustering using Hierarchical (n_clust={n_clusters})')
    plt.legend()

    plt.savefig(FIGURE_PATH + f"scaled_trajectories/feature_clustering_hierarchical_{n_clusters}_clusters.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    
# ## DTW-based clustering

# # Prepare variable-length sequences
# variable_length_seqs = []
# for i in tqdm(list(range(n_patients))):
#     row = px_trajectories[i]
#     # drop NaNs at the end
#     last_idx = np.where(np.isnan(row))[0]
#     if len(last_idx) == 0:
#         seq = row
#     else:
#         seq = row[:last_idx[0]]

#     # handle empty / all-NaN case
#     if len(seq) == 0 or np.all(np.isnan(seq)):
#         seq = np.array([0.0])  # fallback dummy sequence

#     variable_length_seqs.append(seq)

# # Downsample
# def downsample(seq, factor=2):
#     """Downsample sequence by averaging every `factor` points."""
#     if len(seq) < factor:
#         return seq
#     n = (len(seq) // factor) * factor
#     return np.mean(seq[:n].reshape(-1, factor), axis=1)

# downsample_factor = 2   # try 2 or 4
# variable_length_seqs = [downsample(seq, downsample_factor) for seq in variable_length_seqs]

# # Compute DTW distance matrix
# dist_matrix = cdist_dtw(variable_length_seqs)   # full n x n matrix

# # Hierarchical clustering
# Z = linkage(squareform(dist_matrix), method='average')

# for n_clusters in range(2,11):

#     cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

#     plt.figure(figsize=(14,8))
#     for cl in range(1,n_clusters+1):
#         members = px_trajectories[cluster_labels == cl]
#         mean_traj = np.nanmean(members, axis=0)
#         lower = np.nanpercentile(members, 2.5, axis=0)
#         upper = np.nanpercentile(members, 97.5, axis=0)
#         plt.plot(time, mean_traj, label=f"Cluster {cl} mean", linewidth=3)
#         plt.fill_between(time, lower, upper, alpha=0.2)

#     plt.xlabel("Timepoint")
#     plt.ylabel("Risk Score")
#     plt.title(f"Hierarchical Clustering of Variable-Length Risk Trajectories via DTW (n_clust={n_clusters})")
#     plt.legend()
    
#     plt.savefig(FIGURE_PATH + f"dtw_clustering_w_{n_clusters}_clusters.png", dpi=300, bbox_inches='tight')
#     plt.close()