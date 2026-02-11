import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr  
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Paths
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
FIGURE_PATH = os.path.join(PROJ_PATH, 'figures/model_metrics/')
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
FEATURE_PATH = os.path.join(DATA_PATH, 'clinical_and_genomic_features/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/icd_results/')
TRAJECTORY_PATH = os.path.join(RESULTS_PATH, 'mortality_trajectories/')

FIGURE_PATH = os.path.join(PROJ_PATH, 'figures/mortality_trajectories/')
FEATURE_FIG_PATH = os.path.join(FIGURE_PATH, 'feature_based_clusters_decay_param=0.01/')
UNSCALED_FEATURE_FIG_PATH = os.path.join(FEATURE_FIG_PATH, 'unscaled/')
ADJUSTED_FEATURE_FIG_PATH = os.path.join(FEATURE_FIG_PATH, 'adjusted/')

os.makedirs(FEATURE_FIG_PATH, exist_ok=True)
os.makedirs(UNSCALED_FEATURE_FIG_PATH, exist_ok=True)
os.makedirs(ADJUSTED_FEATURE_FIG_PATH, exist_ok=True)

def zscore(ts):
    return (ts - ts.mean()) / (ts.std() + 1e-8)

def gen_clusters(feature_df, feature_cols, clusters_to_test=[i+2 for i in range(18)]):
    inertias = []
    silhouette_scores = []
    cluster_labels = {'DFCI_MRN' : feature_df['DFCI_MRN']}
    for n_clust in tqdm(clusters_to_test):
        km = KMeans(n_clusters=n_clust, random_state=0).fit(feature_df[feature_cols])
        clusters = km.predict(feature_df[feature_cols])
        inertias.append(km.inertia_)
        cluster_labels[f'label_w_{n_clust}_clusters'] = clusters

    cluster_label_df = pd.DataFrame(cluster_labels)
    cluster_label_df[[col for col in cluster_label_df.columns if col != 'DFCI_MRN']] += 1
    return cluster_label_df, inertias

def gen_mean_risk_traj_plot(long_traj_df, output_path, x='time', y='risk_score', title = 'Mean Risk Trajectories by Cluster', figsize=(10,6)):
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=long_traj_df,
        x=x,
        y=y,
        hue='cluster_label',
        estimator='mean',
        errorbar='sd',
        linewidth=3)
    
    plt.title(title)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def gen_spaghetti_plot(long_traj_df, output_path, x='time', y='risk_score', title='Risk Trajectories by Cluster'):
    g=sns.relplot(
        data=long_traj_df,
        x=x,
        y=y,
        col='cluster_label',
        kind='line',
        units='DFCI_MRN',
        estimator=None,
        alpha=0.2,
        linewidth=1,
        col_wrap=2,
        height=3,
        aspect=1.2)
    
    g.set_axis_labels(x, y)
    g.fig.suptitle(title, y=1.05)
    plt.savefig(output_path)
    plt.close()
    
def gen_trajectory_heatmap(long_traj_df, output_path, x='time', y='risk_score', title='Risk Score Heatmap Ordered by Cluster', figsize=(10,8)):
    heatmap_df = (long_traj_df
                  .pivot_table(index='DFCI_MRN',
                               columns=x,
                               values=y))
    
    order = (long_traj_df
             .groupby('DFCI_MRN')
             .agg(cluster=('cluster_label', 'first'),
                  mean_risk=(y, 'mean'))
             .sort_values(['cluster', 'mean_risk'])
             .index)
    
    ordered_clusters = (
        long_traj_df
        .groupby('DFCI_MRN')['cluster_label']
        .first()
        .loc[order])
    
    cluster_sizes = ordered_clusters.value_counts(sort=False)
    cluster_starts = np.cumsum([0] + cluster_sizes.tolist()[:-1])
    cluster_mids = cluster_starts + cluster_sizes.values /2
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heatmap_df.loc[order],
        cmap='vlag',
        center=0,
        cbar_kws={'label' : y},
        yticklabels=False)
    
    ax.set_yticks(cluster_mids)
    ax.set_yticklabels([f'Cluster {c}' for c in cluster_sizes.index], rotation=0)
    
    for y in cluster_starts[1:]:
        ax.hlines(y, *ax.get_xlim(), colors='black', linewidth=0.5)
        
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    
# Load datasets
stage_df = pd.read_csv(os.path.join(FEATURE_PATH, 'cancer_stage_df.csv'))
type_df = pd.read_csv(os.path.join(FEATURE_PATH, 'cancer_type_df.csv'))

stage_cols = ["CANCER_STAGE_2.0", "CANCER_STAGE_3.0", "CANCER_STAGE_4.0"]
stage_df["STAGE"] = (stage_df[stage_cols]
                     .mul([2, 3, 4])
                     .sum(axis=1)
                     .replace(0, 1))
stage_df.drop(columns=stage_cols, inplace=True)

# surv_traj = pd.read_csv(os.path.join(TRAJECTORY_PATH, 'survival_trajectories.csv'))
surv_traj = pd.read_csv(os.path.join(TRAJECTORY_PATH, 'survival_trajectories_w_decay_param_0.1.csv'))

traj_w_stage = surv_traj.merge(stage_df, on='DFCI_MRN')[['DFCI_MRN', 'STAGE', 'plus_0_months_data']].dropna()
rho, p = spearmanr(traj_w_stage['STAGE'], traj_w_stage['plus_0_months_data'])
print(f'Spearman\'s rho between stage and risk score at time 0 = {rho : 0.2f}')
print(f'p-value on the Spearman\'s rho = {p : 0.3f}')

traj_long = (surv_traj.loc[~surv_traj['plus_0_months_data'].isna() & ~surv_traj['plus_3_months_data'].isna()]
             .melt(id_vars='DFCI_MRN',
                   value_vars = [c for c in surv_traj.columns if c.startswith('plus_')],
                   var_name='time',
                   value_name='risk_score')
             .dropna())
traj_long['time'] = traj_long['time'].apply(lambda x : int(x.split('_')[1]))
traj_long = traj_long.sort_values(by=['DFCI_MRN', 'time'])

features = (
    traj_long
    .groupby("DFCI_MRN")
    .apply(lambda x: pd.Series({
        "baseline": x.sort_values("time").iloc[0]["risk_score"],
        "slope": np.polyfit(x["time"], x["risk_score"], 1)[0],
        "auc": np.trapz(x["risk_score"], x["time"]),
        "max": x["risk_score"].max()}))
    .reset_index())

cluster_label_df, inertias = gen_clusters(features, [col for col in features.columns if col != 'DFCI_MRN'])

chosen_clust_num=4
plt.plot(range(len(inertias)), inertias)
plt.axvline(x=chosen_clust_num, color='red', label=f'{chosen_clust_num} clusters')
plt.title('Raw Mortality Trajectory Elbow Plot')
plt.legend()
plt.savefig(os.path.join(UNSCALED_FEATURE_FIG_PATH, 'elbow_plot.png'))
plt.close()

trajs_to_plot = (traj_long
                 .merge(cluster_label_df[['DFCI_MRN', f'label_w_{chosen_clust_num}_clusters']], on='DFCI_MRN')
                 .rename(columns={f'label_w_{chosen_clust_num}_clusters' : 'cluster_label'})
                 .sort_values(by=['cluster_label', 'DFCI_MRN', 'time']))

gen_mean_risk_traj_plot(trajs_to_plot, output_path=os.path.join(UNSCALED_FEATURE_FIG_PATH, 'mean_risk_trajectories_by_cluster.png'))
gen_spaghetti_plot(trajs_to_plot, output_path=os.path.join(UNSCALED_FEATURE_FIG_PATH, 'spaghetti_plot_by_cluster.png'))
gen_trajectory_heatmap(trajs_to_plot, output_path=os.path.join(UNSCALED_FEATURE_FIG_PATH, 'heatmap_by_cluster.png'))

mean_risk_score_by_cluster = (trajs_to_plot
                              .groupby('cluster_label')['risk_score']
                              .mean()
                              .reset_index()
                              .sort_values(by='risk_score'))
mean_risk_score_by_cluster['ordered_labels'] = range(len(mean_risk_score_by_cluster))
updated_label_dict = dict(zip(mean_risk_score_by_cluster['cluster_label'],
                              mean_risk_score_by_cluster['ordered_labels']))

cluster_stage_df = (cluster_label_df[['DFCI_MRN', f'label_w_{chosen_clust_num}_clusters']]
                    .merge(stage_df[['DFCI_MRN', 'STAGE']], on='DFCI_MRN')
                    .rename(columns={f'label_w_{chosen_clust_num}_clusters' : 'cluster_label'}))
cluster_stage_df['ordered_cluster_labels'] = cluster_stage_df['cluster_label'].map(updated_label_dict)
rho, p = spearmanr(cluster_stage_df['ordered_cluster_labels'], cluster_stage_df['STAGE'])
print(f'Spearman rho = {rho : 0.2f}')
print(f'p-value = {p : 0.3f}')

risk_cols = [c for c in surv_traj.columns if c.startswith('plus_')]

adjusted_surv_traj = (surv_traj.loc[~surv_traj['plus_0_months_data'].isna() & ~surv_traj['plus_3_months_data'].isna()]).copy()
adjusted_surv_traj[risk_cols] = adjusted_surv_traj[risk_cols] - adjusted_surv_traj['plus_0_months_data'].values.reshape(-1,1)

adjusted_traj_long = (adjusted_surv_traj
                      .melt(id_vars='DFCI_MRN', 
                            value_vars=risk_cols, 
                            var_name='time', 
                            value_name='risk_score').dropna())

adjusted_traj_long['time'] = adjusted_traj_long['time'].apply(lambda x : int(x.split('_')[1]))
adjusted_traj_long = adjusted_traj_long.sort_values(by=['DFCI_MRN', 'time'])

adjusted_features = (
    adjusted_traj_long
    .groupby("DFCI_MRN")
    .apply(lambda x: pd.Series({
        "baseline": x.sort_values("time").iloc[0]["risk_score"],
        "slope": np.polyfit(x["time"], x["risk_score"], 1)[0],
        "auc": np.trapz(x["risk_score"], x["time"]),
        "max": x["risk_score"].max()}))
    .reset_index())

adjusted_cluster_label_df, adjusted_inertias = gen_clusters(adjusted_features, [col for col in features.columns if col != 'DFCI_MRN'])

chosen_clust_num=4
plt.plot(range(len(adjusted_inertias)), adjusted_inertias)
plt.axvline(x=chosen_clust_num, color='red', label=f'{chosen_clust_num} clusters')
plt.title('Adjusted Mortality Trajectory Elbow Plot')
plt.legend()
plt.savefig(os.path.join(ADJUSTED_FEATURE_FIG_PATH, 'elbow_plot.png'))
plt.close()

trajs_to_plot = (adjusted_traj_long
                 .merge(adjusted_cluster_label_df[['DFCI_MRN', f'label_w_{chosen_clust_num}_clusters']], on='DFCI_MRN')
                 .rename(columns={f'label_w_{chosen_clust_num}_clusters' : 'cluster_label'})
                 .sort_values(by=['cluster_label', 'DFCI_MRN', 'time']))
gen_mean_risk_traj_plot(trajs_to_plot, output_path=os.path.join(ADJUSTED_FEATURE_FIG_PATH, 'mean_c_index_trajectories_by_cluster.png'))
gen_spaghetti_plot(trajs_to_plot, output_path=os.path.join(ADJUSTED_FEATURE_FIG_PATH, 'spaghetti_plot_by_cluster.png'))
gen_trajectory_heatmap(trajs_to_plot, output_path=os.path.join(ADJUSTED_FEATURE_FIG_PATH, 'heatmap_by_cluster.png'))